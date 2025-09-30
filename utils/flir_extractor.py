#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function

import sys
import glob
import argparse
import io
import json
import os
import os.path
import re
import csv
import subprocess
from PIL import Image
from math import sqrt, exp, log
from matplotlib import cm
from matplotlib import pyplot as plt

import cv2 as cv
import pandas as pd
from itertools import zip_longest

import numpy as np


class FlirImageExtractor:

    def __init__(self, exiftool_path="exiftool", is_debug=False, provided_metadata=None):
        self.exiftool_path = exiftool_path
        self.is_debug = is_debug
        self.extracted_metadata = None
        self.provided_metadata= provided_metadata
        self.updated_metadata = None

        self.flir_img_filename = ""
        self.use_thumbnail = False
        self.fix_endian = True

        self.rgb_image_np = None
        self.cropped_visual_np = None
        self.thermal_image_np = None

    def extract_metadata(self, flir_img_filename):
        self.flir_img_filename = flir_img_filename
        if self.is_debug:
            print("DEBUG: Extracting metadata from Flir image in filepath:{}".format(flir_img_filename))

        if not os.path.isfile(flir_img_filename):
            raise ValueError("Input file does not exist or permission denied")

        meta_json = subprocess.check_output(
            [self.exiftool_path, self.flir_img_filename, '-Emissivity', '-SubjectDistance', '-AtmosphericTemperature',
            '-ReflectedApparentTemperature', '-IRWindowTemperature', '-IRWindowTransmission', '-RelativeHumidity',
            '-PlanckR1', '-PlanckB', '-PlanckF', '-PlanckO', '-PlanckR2', '-j'])
        meta = json.loads(meta_json.decode())[0]
        return meta
    

    def modify_metadata(self, flir_img_filename):
        self.extracted_metadata = self.extract_metadata(flir_img_filename)

        if self.extracted_metadata and self.provided_metadata:
            self.updated_metadata = {k: self.provided_metadata.get(k, v) for k, v in self.extracted_metadata.items()}
            if self.is_debug:
                print("DEBUG: Updated Metadata:{}".format(self.updated_metadata))
            return self.updated_metadata
    

    def process_image(self, flir_img_filename):
        if self.is_debug:
            print("DEBUG: Will reconstruct images and generate temperatures for Flir image with filepath:{}".format(flir_img_filename))
            
        print("Processing...")
        if not os.path.isfile(flir_img_filename):
            raise ValueError("Input file does not exist or permission denied")

        self.flir_img_filename = flir_img_filename
        self.rgb_image_np = self.extract_embedded_image()
        self.thermal_image_np = self.extract_thermal_image()

    def get_rgb_np(self):
        return self.rgb_image_np

    def get_thermal_np(self):
        return self.thermal_image_np

    def extract_embedded_image(self):
        image_tag = "-EmbeddedImage"
        print("Extracting the visual image")

        visual_img_bytes = subprocess.check_output([self.exiftool_path, image_tag, "-b", self.flir_img_filename])
        visual_img_stream = io.BytesIO(visual_img_bytes)

        visual_img = Image.open(visual_img_stream)
        visual_np = np.array(visual_img)
        return visual_np

    def extract_thermal_image(self):
        meta = self.updated_metadata
        thermal_img_bytes = subprocess.check_output([self.exiftool_path, "-RawThermalImage", "-b", self.flir_img_filename])
        thermal_img_stream = io.BytesIO(thermal_img_bytes)

        thermal_img = Image.open(thermal_img_stream)
        thermal_np = np.array(thermal_img)

        if self.fix_endian:
            thermal_np = np.vectorize(lambda x: (x >> 8) + ((x & 0x00ff) << 8))(thermal_np)

        raw2tempfunc = np.vectorize(lambda x: FlirImageExtractor.raw2temp(x,
                                                                E=FlirImageExtractor.extract_float(meta['Emissivity']), 
                                                                OD=FlirImageExtractor.extract_float(meta['SubjectDistance']),
                                                                RTemp=FlirImageExtractor.extract_float(meta['ReflectedApparentTemperature']),
                                                                ATemp=FlirImageExtractor.extract_float(meta['AtmosphericTemperature']),
                                                                IRWTemp=FlirImageExtractor.extract_float(meta['IRWindowTemperature']),
                                                                IRT=meta['IRWindowTransmission'],
                                                                RH=FlirImageExtractor.extract_float(meta['RelativeHumidity']),
                                                                PR1=meta['PlanckR1'], PB=meta['PlanckB'],
                                                                PF=meta['PlanckF'],
                                                                PO=meta['PlanckO'], PR2=meta['PlanckR2']))
        thermal_np = raw2tempfunc(thermal_np)
        return thermal_np

    @staticmethod
    def raw2temp(raw, E=0.98, OD=15.24, RTemp=30, ATemp=20, IRWTemp=20, IRT=1, RH=50,
                 PR1=21106.77, PB=1501, PF=1, PO=-7340, PR2=0.012545258):
        ATA1 = 0.006569
        ATA2 = 0.01262
        ATB1 = -0.002276
        ATB2 = -0.00667
        ATX = 1.9

        emiss_wind = 1 - IRT
        refl_wind = 0

        h2o = (RH / 100) * exp(1.5587 + 0.06939 * (ATemp) - 0.00027816 * (ATemp) ** 2 + 0.00000068455 * (ATemp) ** 3)
        tau1 = ATX * exp(-sqrt(OD / 2) * (ATA1 + ATB1 * sqrt(h2o))) + (1 - ATX) * exp(-sqrt(OD / 2) * (ATA2 + ATB2 * sqrt(h2o)))
        tau2 = ATX * exp(-sqrt(OD / 2) * (ATA1 + ATB1 * sqrt(h2o))) + (1 - ATX) * exp(-sqrt(OD / 2) * (ATA2 + ATB2 * sqrt(h2o)))

        raw_refl1 = PR1 / (PR2 * (exp(PB / (RTemp + 273.15)) - PF)) - PO
        raw_refl1_attn = (1 - E) / E * raw_refl1
        raw_atm1 = PR1 / (PR2 * (exp(PB / (ATemp + 273.15)) - PF)) - PO
        raw_atm1_attn = (1 - tau1) / E / tau1 * raw_atm1
        raw_wind = PR1 / (PR2 * (exp(PB / (IRWTemp + 273.15)) - PF)) - PO
        raw_wind_attn = emiss_wind / E / tau1 / IRT * raw_wind
        raw_refl2 = PR1 / (PR2 * (exp(PB / (RTemp + 273.15)) - PF)) - PO
        raw_refl2_attn = refl_wind / E / tau1 / IRT * raw_refl2
        raw_atm2 = PR1 / (PR2 * (exp(PB / (ATemp + 273.15)) - PF)) - PO
        raw_atm2_attn = (1 - tau2) / E / tau1 / IRT / tau2 * raw_atm2
        raw_obj = (raw / E / tau1 / IRT / tau2 - raw_atm1_attn -
                   raw_atm2_attn - raw_wind_attn - raw_refl1_attn - raw_refl2_attn)

        temp_celcius = PB / log(PR1 / (PR2 * (raw_obj + PO)) + PF) - 273.15
        return temp_celcius

    @staticmethod
    def extract_float(dirtystr):
        digits = re.findall(r"[-+]?\d*\.\d+|\d+", dirtystr)
        return float(digits[0])

    def save_images(self):
        rgb_np = self.get_rgb_np()
        thermal_np = self.thermal_image_np

        img_visual = Image.fromarray(rgb_np)
        self.cropped_visual_np = crop_image_only_outside(rgb_np, 30)
        cropped_img_visual = Image.fromarray(self.cropped_visual_np)

        widthDiff = img_visual.size[0] - cropped_img_visual.size[0]
        heightDiff = img_visual.size[1] - cropped_img_visual.size[1]

        thermal_normalized = (thermal_np - np.amin(thermal_np)) / (np.amax(thermal_np) - np.amin(thermal_np))
        img_thermal = Image.fromarray(np.uint8(cm.inferno(thermal_normalized) * 255))

        fn_prefix, _ = os.path.splitext(self.flir_img_filename)
        thermal_image_path = os.path.join(fn_prefix.replace('Flir_Images','Thermal_Images')+'.png')
        visual_image_path = os.path.join(fn_prefix.replace('Flir_Images','Visual_Images')+'.jpg')
        visual_image_nocrop_path = os.path.join(fn_prefix.replace('Flir_Images','Visual_Images_nocrop')+'.jpg')

        img_visual.save(visual_image_nocrop_path)
        cropped_img_visual.save(visual_image_path)
        img_thermal.save(thermal_image_path)

        flat_thermal_np = thermal_np.flatten()
        minTemp = min(flat_thermal_np)
        maxTemp = max(flat_thermal_np)
        return widthDiff, heightDiff, thermal_np, minTemp, maxTemp

    def export_data_to_csv(self):
        fn_prefix, _ = os.path.splitext(self.flir_img_filename)
        csv_path = os.path.join(fn_prefix.replace('Flir_Images','Csv_Files')+'.csv')
        
        downscaled_visual_np = image_downscale(self.cropped_visual_np, 80, 60)
        coords_and_thermal_values = []
        for e in np.ndenumerate(self.thermal_image_np):
            x, y = e[0]
            c = e[1]
            coords_and_thermal_values.append([x, y, c])
    
        rgb_values = []
        for i in range(downscaled_visual_np.shape[0]):
            for j in range(downscaled_visual_np.shape[1]):
                R = downscaled_visual_np[i,j,0]
                G = downscaled_visual_np[i,j,1]
                B = downscaled_visual_np[i,j,2]
                rgb_values.append([R, G, B])
        
        merged_list = list(map(list,zip(coords_and_thermal_values, rgb_values)))
        flat_list = [item for sublist in merged_list for item in sublist]
        x = iter(flat_list)
        formatted_flat_list = [a+b for a, b in zip_longest(x, x, fillvalue=[])]
        
        with open(csv_path, 'w') as fh:
            writer = csv.writer(fh, delimiter=',')
            writer.writerow(['x', 'y', 'Temp(c)', 'R', 'G', 'B'])
            writer.writerows(formatted_flat_list)


# ================== UTILITY FUNCTIONS ==================

def image_downscale(img_np, width, height):
    dim = (width, height)
    resized_visual_np = cv.resize(img_np, dim, interpolation=cv.INTER_AREA)
    return resized_visual_np


def crop_image_only_outside(img_np, tol=0):
    mask = img_np > tol
    if img_np.ndim == 3:
        mask = mask.all(2)
    m, n = mask.shape
    mask0, mask1 = mask.any(0), mask.any(1)
    col_start, col_end = mask0.argmax(), n - mask0[::-1].argmax()
    row_start, row_end = mask1.argmax(), m - mask1[::-1].argmax()
    return img_np[row_start:row_end, col_start:col_end]


def crop_center(img, cropx, cropy):
    y, x, z = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx]


def crop_mask_and_overlay_temps(temps_np, mask_path, crop_w, crop_h, at=0, val_sub=0, val_add=0):
    mask_visual = Image.open(mask_path)
    mask_np = np.asarray(mask_visual)

    initial_mask_width = mask_np.shape[1]
    initial_mask_height = mask_np.shape[0]
    print(f"DEBUG: Original mask width: {initial_mask_width}, height: {initial_mask_height}")
    print(f"Width to remove: {crop_w}, Height to remove: {crop_h}")

    final_mask_width = initial_mask_width - crop_w
    final_mask_height = initial_mask_height - crop_h
    mask_np = crop_center(mask_np, final_mask_width, final_mask_height)

    mask_np_visual = Image.fromarray(mask_np)
    mask_np_visual.save(mask_path, 'PNG')

    not_leaves_mask = np.int64(np.all(mask_np[:, :, :3] == 0, axis=2))
    downscaled_not_leaves_mask = cv.resize(np.uint8(not_leaves_mask), dsize=(80, 60), interpolation=cv.INTER_CUBIC)

    threshold_min = at - val_sub
    threshold_max = at + val_add
    thermal_thresholding_mask = np.int64(np.logical_or(temps_np < threshold_min, temps_np > threshold_max))

    final_exclusion_mask_np = thermal_thresholding_mask | downscaled_not_leaves_mask
    temps_np_masked = np.ma.masked_array(temps_np, mask=final_exclusion_mask_np, fill_value=999)

    sunlit_leaves_only = temps_np_masked[~temps_np_masked.mask]
    sunlit_leaves_mean_temp = sunlit_leaves_only.mean()

    return sunlit_leaves_mean_temp, temps_np_masked.filled()


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def calculateCWSI(Ta, Tc, RH):
    Slope = -1.49
    Intercept = 3.09

    VPsat = 0.6108 * math.exp(17.27 * Ta / (Ta + 237.3))
    VPair = VPsat * RH/100
    VPD = VPsat - VPair
    VPsat_Ta_plus_Intercept = 0.6108 * math.exp(17.27 * (Ta + Intercept) / (Ta + Intercept + 237.3))
    VPG = VPsat - VPsat_Ta_plus_Intercept

    T_ll = Intercept + Slope * VPD
    T_ul = Intercept + Slope * VPG

    CWSI = ((Tc - Ta) - T_ll) / (T_ul - T_ll)
    return CWSI

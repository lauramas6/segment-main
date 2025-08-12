MODEL_ZOO = {
    "frrn_a": {
        "default_model": None,   # local impl
        "num_classes": 2,
        "in_channels": 3,
        "trust_remote_code": False,
        "image_size": (512, 512),
        "aliases": ["frrn"],
    },
    "fc_densenet103": {
        "default_model": None,   # local impl
        "num_classes": 2,
        "in_channels": 3,
        "trust_remote_code": False,
        "image_size": (512, 512),
        "aliases": ["fcdensenet", "tiramisu"],
    },
    "segformer": {
        "default_model": "nvidia/segformer-b3-finetuned-ade-512-512",
        "num_classes": 150,
        "in_channels": 3,
        "trust_remote_code": False,
        "image_size": (512, 512),
    },
    "setr": {
        "default_model": "damo/SETR_MLA",
        "num_classes": 150,
        "in_channels": 3,
        "trust_remote_code": True,
        "image_size": (512, 512),
    },
    "mask2former": {
        "default_model": "shi-labs/mask2former-swin-large-ade",
        "num_classes": 150,
        "in_channels": 3,
        "trust_remote_code": True,
        "image_size": (512, 512),
    },
    "deeplabv3": {
        "default_model": "torchvision/deeplabv3-resnet50",
        "num_classes": 21,
        "in_channels": 3,
        "trust_remote_code": False,
        "image_size": (520, 520),
    },
    "fcn": {
        "default_model": "torchvision/fcn-resnet50",
        "num_classes": 21,
        "in_channels": 3,
        "trust_remote_code": False,
        "image_size": (520, 520),
    },
    "pspnet": {
        "default_model": "open-mmlab/pspnet_r50-d8_512x512_80k_ade20k",
        "num_classes": 150,
        "in_channels": 3,
        "trust_remote_code": True,
        "image_size": (512, 512),
    },
}

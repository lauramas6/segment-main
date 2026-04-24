MODEL_ZOO = {
    "sam": {
        "default_model": "facebook/sam3",
        "num_classes": 2,             # adjust if multi-class
        "in_channels": 3,             # RGB input
        "trust_remote_code": True,    # SAM needs remote code
        "image_size": (1024, 1024),   # native SAM resolution
        "aliases": ["segment_anything", "samvit", "sam_vit_b"],
    },
    "frrn_a": {
        "default_model": None,   # local impl
        "num_classes": 2,
        "in_channels": 3,
        "trust_remote_code": False,
        "image_size": (512, 512),
        "aliases": ["frrn"],
    },
    "fcdensenet": {
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
        "num_classes": 2,
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
    "mask2former": {
        "default_model": "facebook/mask2former-swin-small-ade-semantic",
        "num_classes": 2,
        "in_channels": 3,
        "trust_remote_code": False,
        "image_size": (518, 518),
    },
    "dinov3": {
        "default_model": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "num_classes": 2,
        "in_channels": 3,
        "trust_remote_code": False,
        "image_size": (518, 518),
        "aliases": ["dinov3b", "dinov3_vitb", "dinov3_base"],
    },
    
    # ADD NEW MODEL REQUIREMENTS HERE!
}

from config import CFG

# Canonical architectures -> module paths
VALID_ARCHS = {
    # ViT-based
    "segformer": "vit.segformer",
    "setr": "vit.setr",
    "sam": "vit.sam",
    "dinov3": "vit.dinov3",
    "mask2former": "vit.mask2former",

    # CNN-based
    "deeplabv3": "cnn.deeplabv3",
    "fcn": "cnn.fcn",
    "pspnet": "cnn.pspnet",

    # New locals
    "frrn_a": "cnn.frrn_a",
    "fc_densenet103": "cnn.fcdensenet",
}

# Aliases
ALIASES = {
    "frrn": "frrn_a",
    "tiramisu": "fc_densenet103",
    "fc_densenet": "fc_densenet103",
    "fcdensenet": "fc_densenet103",
    "m2f": "mask2former",
    "dinov3b": "dinov3",
    "dinov3_vitb": "dinov3",
    "dinov3_base": "dinov3",
}

def _resolve_arch(name: str) -> str:
    n = name.lower()
    return ALIASES.get(n, n)

def get_model():
    arch = _resolve_arch(CFG.architecture)
    if arch not in VALID_ARCHS:
        allowed = sorted(set(list(VALID_ARCHS.keys()) + list(ALIASES.keys())))
        raise ValueError(
            f"Invalid architecture '{CFG.architecture}'. Must be one of: {allowed}"
        )

    module_path = VALID_ARCHS[arch]
    module = __import__(f"models.{module_path}", fromlist=["get_model_func"])

    func_name = f"get_{arch}_model"
    if hasattr(module, func_name):
        return getattr(module, func_name)(CFG)
    else:
        raise ImportError(f"`{func_name}()` not found in {module_path}.py")

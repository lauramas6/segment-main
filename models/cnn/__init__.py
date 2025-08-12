"""
CNN model implementations for semantic segmentation.

Each file in this package should implement:
    def get_<architecture>_model(CFG):
        -> returns a torch.nn.Module ready for training/evaluation.

Expected filenames and entry points:
    - deeplabv3.py       -> get_deeplabv3_model()
    - fcn.py             -> get_fcn_model()
    - pspnet.py          -> get_pspnet_model()
    - frrn_a.py          -> get_frrn_a_model()
    - fcdensenet.py      -> get_fc_densenet103_model()
"""

# Optional: expose available architectures here (not required if using dynamic imports)
__all__ = [
    "deeplabv3",
    "fcn",
    "pspnet",
    "frrn_a",
    "fcdensenet",
]

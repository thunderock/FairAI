# @Filename:    config_utils.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        7/8/22 4:49 PM
import sys


def get_sk_param(param, default=None, param_idx=None):
    if param_idx is not None:
        return sys.argv[param_idx]
    if "snakemake" in sys.modules:
        return snakemake.params.get(param, default)
    return default
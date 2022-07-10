import numpy as np
from utils.config_utils import get_sk_value, CONSTANTS

val = get_sk_value(param="device", field=snakemake.params)
print("testing string: ", type(val), val)
val = get_sk_value(param="checkpoint", field=snakemake.params)
print("testing int: ", type(val), val)
val = get_sk_value(param=CONSTANTS.DIST_METRIC.__name__.lower(), field=snakemake.params, object=True)
print("testing object: ", type(val), val)
val = get_sk_value(param="lr", field=snakemake.params)
print("testing float: ", type(val), val)

np.save("/tmp/scores.npy", np.zeros(1))
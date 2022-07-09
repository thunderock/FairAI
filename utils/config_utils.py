# @Filename:    config_utils.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        7/8/22 4:49 PM
import sys
import gravlearn

def _get_sk_value(param, sk_field, default=None, object=False, type=None):
    assert not (object and type), "Cannot specify both object and type"
    val = default
    try:
        val = sk_field[param]
    except:
        print("Could not find parameter: ", param, " assigning default: ", default)
    if object:
        print("reached here ", val, sk_field, param)
        return _OBJECT_MAP.get(val, val)
    if type is not None:
        return type(val)
    return val

def get_sk_value(param, field, default=None, object=False, type=None):
    if "snakemake" in sys.modules:
        return _get_sk_value(param, sk_field=field, default=default, object=object, type=type)
    return default

def get_input_value(param_idx, default=None, type=None):
    val = default
    try:
        val = sys.argv[param_idx]
    except IndexError:
        pass
    if type is not None:
        return type(val)
    return val


class CONSTANTS(object):
    class DIST_METRIC(object):
        DOTSIM = 'dotsim'
    DEVICE = "device"


_OBJECT_MAP = {
    CONSTANTS.DIST_METRIC.DOTSIM: gravlearn.metrics.DistanceMetrics.DOTSIM
}
# @Filename:    config_utils.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        7/8/22 4:49 PM
import sys
import gravlearn

# TODO (ashutiwa): remove all print statemenst with logging
def set_snakemake_config(param, value, field_name="params"):
    if field_name not in _CONFIG_STORE:
        _CONFIG_STORE[field_name] = {}
    _CONFIG_STORE[field_name][param] = value


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
    elif field in _CONFIG_STORE:
        return _CONFIG_STORE[field].get(param, default)
    print("could not find the parameter filter!! ")
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
    PARAMS = "params"
    OUTPUT = "output"


_OBJECT_MAP = {
    CONSTANTS.DIST_METRIC.DOTSIM: gravlearn.metrics.DistanceMetrics.DOTSIM
}

_CONFIG_STORE = {

}
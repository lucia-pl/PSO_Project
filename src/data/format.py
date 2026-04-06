import numpy as np

def to_python_type(value):
    """
    Turns a given value whic format is based on numpy to a format native from python

    Args:
        value: needed to be converted

    Returns:
        Converted value or the value itself if it was never np. based
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [to_python_type(v) for v in value]
    return value
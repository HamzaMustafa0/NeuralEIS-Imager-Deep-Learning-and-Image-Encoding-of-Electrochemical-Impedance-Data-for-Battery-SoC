import numpy as np
from eis2img.transforms.gaf import GAFTransformer

def test_gaf_shape_and_dtype():
    N = 32
    r = np.linspace(0, 1, N)
    x = np.linspace(-0.5, 0.5, N)
    img = GAFTransformer().encode_rgb(r, x)
    assert img.shape == (N, N, 3)
    assert img.dtype == np.uint8
    assert img.min() >= 0 and img.max() <= 254

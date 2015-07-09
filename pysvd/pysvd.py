import logging

import numpy as np
from numpy.linalg import svd
import scipy.misc
from PIL import Image
import click

class PySVDValueError(ValueError):
    pass

def truncate_svd_factors_n_largest(u, s, v, n=True):
    if n is True:
        n = len(s)
    elif n < 1 or int(n) != n:
        raise PySVDValueError("n must be a positive integer")

    u_trunc = np.delete(u, np.s_[n:], 1) # cut [n:] columns
    s_trunc = np.delete(s, np.s_[n:]) # cut [n:] singular values
    v_trunc = np.delete(v, np.s_[n:], 0) # cut [n:] rows

    return u_trunc, s_trunc, v_trunc

def truncate_svd_factors_n_largest_frac(u, s, v, n=True):
    """n is a value (0,1] indicating the fraction of the largest singular
    values. n may also be True, meaning all singular values are kept.
    A truncation to 0 makes little sense, so this function will always return
    at least 1 singular value.
    """
    if n is True:
        return truncate_svd_factors_n_largest(u, s, v, n)

    if n > 1.0 or n <= 0.0:
        raise PySVDValueError("frac is too large, must be (0, 1]")
    vals_to_keep = min(len(s) * n, 1)
    return truncate_svd_factors_n_largest(u, s, v, vals_to_keep)

def compose_svd_factors(u, s, v):
    sing_matrix = np.diag(s)
    diff = v.shape[0] - sing_matrix.shape[1]
    if diff > 0:
        extra_zero_cols = np.zeros((sing_matrix.shape[0],diff))
        sing_matrix = np.concatenate((sing_matrix, extra_zero_cols), axis=1)
    return np.dot(u, np.dot(sing_matrix, v))

class Colorspace:
    def __init__(self, name, np_dtype, desc):
        self.name, self.np_dtype, self.desc = name, np_dtype, desc

colorspaces = [
    Colorspace("1", np.dtype('b'), "1-bit pixels, black and white, stored with one pixel per byte"),
    Colorspace("L", np.dtype('i'), "8-bit pixels, black and white"),
    #Colorspace("P", "8-bit pixels, mapped to any other mode using a color palette"),
    Colorspace("RGB", np.dtype('i'), "3x8-bit pixels, true color"),
    Colorspace("RGBA", np.dtype('i'), "4x8-bit pixels, true color with transparency mask"),
    Colorspace("CMYK", np.dtype('i'), "4x8-bit pixels, color separation"),
    Colorspace("YCbCr", np.dtype('i'), "3x8-bit pixels, color video format"),
    Colorspace("LAB", np.dtype('i'), "3x8-bit pixels, the L*a*b color space"),
    Colorspace("HSV", np.dtype('i'), "3x8-bit pixels, Hue, Saturation, Value color space"),
    Colorspace("I", np.dtype('i4'), "32-bit signed integer pixels"),
    Colorspace("F", np.dtype('f4'), "32-bit floating point pixels"),
]

colorspaces_by_name = { cs.name:cs for cs in colorspaces }

@click.command()
@click.argument('input', type=click.File('rb'))
@click.argument('output', type=click.File('wb'))
@click.option('--colorspace', '-c', default='RGB',
    type=click.Choice(colorspaces_by_name.keys()))
@click.option('--nlarge', '-n', default=0, type=click.INT)
@click.option('--fraclarge', '-f', default=0.0, type=click.FLOAT)
def main(input, output, colorspace, nlarge, fraclarge):
    colorspace_obj = colorspaces_by_name[colorspace]

    if not (bool(nlarge) ^ bool(fraclarge)):
        raise click.ClickException("must select ONE of (--nlarge/-n, --fraclarge/-f)")
    elif nlarge:
        func = truncate_svd_factors_n_largest
        n = nlarge
    else:
        func = truncate_svd_factors_n_largest_frac
        n = fraclarge

    image = Image.open(input)
    image = image.convert(mode=colorspace)

    image_shape = image.size

    bands = np.asarray(image)

    if len(bands.shape) > 2:
        depth = bands.shape[2]
        bands = np.dsplit(bands, depth)
        bands = [np.hstack(band).T for band in bands]
    else:
        bands = [bands]

    for i in range(len(bands)):
        band = bands[i]
        factors = svd(band)
        trunced = func(*factors, n=n)
        bands[i] = compose_svd_factors(*trunced).astype(np.uint8)

    if len(bands) > 1:
        bands = np.dstack(bands)
    else:
        bands = bands[0]

    new_image = Image.fromarray(bands, colorspace)
    new_image.save(output)

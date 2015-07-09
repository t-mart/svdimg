import logging
import math
import functools
import operator

import numpy as np
from numpy.linalg import svd
import scipy.misc
from PIL import Image
import click

class PySVDValueError(click.ClickException):
    pass

logger = logging.getLogger(__name__)
main_logger = logging.StreamHandler()
formatter = logging.Formatter('[svdimg] %(message)s')
main_logger.setFormatter(formatter)
logger.addHandler(main_logger)

def truncate_svd_factors_n_largest(u, s, v, n=True):
    if n is True:
        n = len(s)
    elif n < 1 or int(n) != n:
        raise PySVDValueError('n must be a positive integer')

    u_trunc = np.delete(u, np.s_[n:], 1) # cut [n:] columns
    s_trunc = np.delete(s, np.s_[n:]) # cut [n:] singular values
    v_trunc = np.delete(v, np.s_[n:], 0) # cut [n:] rows

    return u_trunc, s_trunc, v_trunc

def truncate_svd_factors_n_largest_frac(u, s, v, n=True):
    '''n is a value (0,1] indicating the fraction of the largest singular
    values. n may also be True, meaning all singular values are kept.
    A truncation to 0 makes little sense, so this function will always return
    at least 1 singular value.
    '''
    if n is True:
        return truncate_svd_factors_n_largest(u, s, v, n)
    if n > 1.0 or n <= 0.0:
        raise PySVDValueError('frac is too large, must be in (0, 1]')
    vals_to_keep = max(int(len(s) * n), 1)
    logger.info('full rank * %f == rank %d', n, vals_to_keep)
    return truncate_svd_factors_n_largest(u, s, v, vals_to_keep)

def find_n_that_equalizes_values(image_shape):
    """For an image of height h and width w, h*w is the number of values in
    this image. A truncated SVD decomposition to rank n of this image will
    produce matrixes of the following sizes: (h,n), (n,1), (n,w), respectively
    the left-singular vector matrix, the singular values (later made into a
    rectangular identity matrix), and the right-singular vector matrix. Summing
    the values of the SVD matrixes is:
        h*n + n + n*w = n(h + 1 + w)

    Therefore, for an SVD to be more efficient in storage than the original
    image, the following must be true

        n(h + 1 + w) < h*w
        or
        n < h*w / (h + 1 + w)

    Note that savings are easiest to obtain when h = w (the image is square).
    For square-ier images, a higher rank can be specified (which improves image
    quality) and still beat h*w.  Conversely, very rectangular images require
    very low ranks (poor quality) to beat h*w in values stored.

    This function produces the minimum rank needed, integer floored, to beat
    h*w in values stored of an SVD decomposition."""

    h, w = image_shape

    return math.floor(h * w / (h + 1 + w))

def compose_svd_factors(u, s, v):
    sing_matrix = np.diag(s)
    diff = v.shape[0] - sing_matrix.shape[1]
    if diff > 0:
        extra_zero_cols = np.zeros((sing_matrix.shape[0],diff))
        sing_matrix = np.concatenate((sing_matrix, extra_zero_cols), axis=1)
    # print('%s . %s . %s' % tuple(str(m.shape) for m in (u, sing_matrix, v)))
    return np.dot(u, np.dot(sing_matrix, v))

class Colorspace:
    def __init__(self, name, np_dtype, clip_interval, desc):
        self.name, self.np_dtype, self.desc, self.clip_interval = name, np_dtype, desc, clip_interval

colorspaces = [
    Colorspace('L', np.uint8, (0, 255), '8-bit pixels, black and white'),
    Colorspace('RGB', np.uint8, (0, 255), '3x8-bit pixels, true color'),
    Colorspace('RGBA', np.uint8, (0, 255), '4x8-bit pixels, true color with transparency mask'),
    Colorspace('CMYK', np.uint8, (0, 255), '4x8-bit pixels, color separation'),
    Colorspace('YCbCr', np.uint8, (0, 255), '3x8-bit pixels, color video format'),
    Colorspace('LAB', np.uint8, (0, 255), '3x8-bit pixels, the L*a*b color space'),
    Colorspace('HSV', np.uint8, (0, 255), '3x8-bit pixels, Hue, Saturation, Value color space'),

    #unsupported PIL modes!
    #======================
    #Colorspace('P', '8-bit pixels, mapped to any other mode using a color palette'),
    #not going to support something that requires a palette argument

    # Colorspace('1', bool, False, '1-bit pixels, black and white, stored with one pixel per byte'),
    # probably a symptom of how SVD works on column/rows, while 1-bit color
    # depth works on radial B/W density, but 1-bit produces unmeaningful
    # results.

    # Colorspace('I', np.dtype('i4'), (0, 2**(32)), '32-bit signed integer pixels'),
    # Colorspace('F', np.dtype('f4'), False, '32-bit floating point pixels'),
    # some error "OSError: cannot write mode I as JPEG", same with F. scrap em'!
]

colorspaces_by_name = { cs.name:cs for cs in colorspaces }

@click.command()
@click.argument('input', type=click.File('rb'))
@click.argument('output', type=click.File('wb'))
@click.option('--colorspace', '-c', default='RGB',
    type=click.Choice(colorspaces_by_name.keys()))
@click.option('--nlarge', '-n', default=0, type=click.INT)
@click.option('--fraclarge', '-f', default=0.0, type=click.FLOAT)
@click.option('--verbose', '-v', count=True)
def main(input, output, colorspace, nlarge, fraclarge, verbose):
    if verbose:
        logger.setLevel('INFO')
        main_logger.setLevel('INFO')

    colorspace_obj = colorspaces_by_name[colorspace]

    if not (bool(nlarge) ^ bool(fraclarge)):
        raise click.ClickException('must select ONE of (--nlarge/-n, --fraclarge/-f)')
    elif nlarge:
        func = truncate_svd_factors_n_largest
        n = nlarge
    else:
        func = truncate_svd_factors_n_largest_frac
        n = fraclarge

    logger.info('opening %s', input.name)
    image = Image.open(input)
    logger.info('converting image to %s', colorspace)
    image = image.convert(mode=colorspace)

    image_shape = image.size
    image_band_names = image.getbands()

    smaller_dim = min(image_shape)
    if n > smaller_dim:
        logger.info('reducing output rank to full rank (%d)', smaller_dim)
        n = smaller_dim

    logger.info('representing image as matrix')
    bands = np.asarray(image).astype(colorspace_obj.np_dtype)

    if bands.ndim > 2:
        logger.info('splitting image channels into separate matrixes %s', str(image_band_names))
        depth = bands.shape[2]
        bands = np.dsplit(bands, depth)
        bands = [np.hstack(band).T for band in bands]
    else:
        bands = [bands]

    band_sizes = []

    for i, name in enumerate(image_band_names):
        band = bands[i]
        logger.info('band[%s] svd factoring', name)
        factors = svd(band)
        logger.info('band[%s] truncating factors to specified rank', name)
        trunced = func(*factors, n=n)
        band_sizes.append([m.shape for m in trunced])
        logger.info('band[%s] multiplying factors back together', name)
        bands[i] = compose_svd_factors(*trunced)
        # print(bands[i][0])
        logger.info('band[%s] clipping color values to 0-255 interval', name)
        if colorspace_obj.clip_interval:
            print(colorspace_obj.clip_interval)
            l, h = colorspace_obj.clip_interval
            np.clip(bands[i], l, h, bands[i]) #since truncated-SVD is an approximation, some values may escape the 0-255 interval
        logger.info('band[%s] converting color values to uint8\'s', name)
        bands[i] = bands[i].astype(colorspace_obj.np_dtype)

    if len(bands) > 1:
        logger.info('merging bands back together')
        bands = np.dstack(bands)
    else:
        bands = bands[0]

    new_image = Image.fromarray(bands, colorspace)
    logger.info('saving image to %s', output.name)
    new_image.save(output)

    #stats printing
    orig_vals = image_shape[0] * image_shape[1] * len(image_band_names)
    logger.info('original image values:')
    logger.info('\t%d = %d*%d*%d',
        orig_vals, image_shape[0], image_shape[1], len(image_band_names))
    svd_vals = 0
    for band in band_sizes:
        svd_vals += (band[0][0] * band[0][1]) + band[1][0] + (band[2][0] * band[2][1])
    band_stat_strs = ['((%d*%d)+%d+(%d*%d))' % (band[0][0], band[0][1], band[1][0], band[2][0], band[2][1])
        for band in band_sizes]
    band_stat_str = " + ".join(band_stat_strs)
    logger.info('svd image values:')
    logger.info('\t%d =  %s', svd_vals, band_stat_str)
    logger.info('svd_vals = %.4f * original_vals', float(svd_vals)/orig_vals)


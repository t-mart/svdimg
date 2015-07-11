import logging
import math
import functools
import operator

import numpy as np
import scipy.linalg
import scipy.misc
from PIL import Image
import click

#set up logging
logger = logging.getLogger(__name__)
main_logger = logging.StreamHandler()
formatter = logging.Formatter('[svdimg] %(message)s')
main_logger.setFormatter(formatter)
logger.addHandler(main_logger)

class Colorspace:
    """A Colorspace stores information about Pillow's Image modes and how they
    interface with numpy."""
    def __init__(self, name, np_dtype, clip_interval, desc):
        """name is a string identifying the mode for Pillow, np_dtype is the
        dtype that numpy should represent an images color data in,
        clip_interval is the interval where color values should remain
        (necessary because multiplying SVD factors may bring some color values
        outside a sensible range, and desc is a string describing the mode."""
        self.name = name
        self.np_dtype = np_dtype
        self.desc = desc
        self.clip_interval = clip_interval


colorspaces = [
    Colorspace('L',     np.uint8, (0, 255), '8-bit pixels, black and white'),
    Colorspace('RGB',   np.uint8, (0, 255), '3x8-bit pixels, true color'),
    Colorspace('RGBA',  np.uint8, (0, 255), '4x8-bit pixels, true color with '
                                            'transparency mask'),
    Colorspace('CMYK',  np.uint8, (0, 255), '4x8-bit pixels, color separation'),
    Colorspace('YCbCr', np.uint8, (0, 255), '3x8-bit pixels, color video '
                                            'format'),
    Colorspace('LAB',   np.uint8, (0, 255), '3x8-bit pixels, the L*a*b color '
                                            'space'),
    Colorspace('HSV',   np.uint8, (0, 255), '3x8-bit pixels, Hue, Saturation, '
                                            'Value color space'),

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

def truncate_svd_factors(u, s, v, rank):
    '''Returns the factors of an SVD u, s, v to rank n. Specifically, u is
    truncated to n columns, s to n singular values, and v to n rows.'''

    u_trunc = u[...,:rank] # remove columns after rank
    s_trunc = s[...,:rank] # remove singular vals after rank
    v_trunc = v[...,:rank,...] # remove rows after rank

    return u_trunc, s_trunc, v_trunc

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
    h*w in values stored of an SVD decomposition. Argument image_shape is a
    tuple of (h, w)."""

    h, w = image_shape

    return math.floor(h * w / (h + 1 + w))

def compose_svd_factors(u, s, v):
    """Multiply SVD factors: u.s.v. These factors should come from numpy's svd
    or or svdimg's truncate_svd_factors function."""
    # build the rectangular diagonal matrix of singular values in s
    # sing_matrix = np.diag(s)
    # diff = v.shape[0] - sing_matrix.shape[1]
    # if diff > 0:
    #     extra_zero_cols = np.zeros((sing_matrix.shape[0],diff))
    #     sing_matrix = np.concatenate((sing_matrix, extra_zero_cols), axis=1)
    if s.ndim > 1:
        sing_matrix = np.asarray([np.diag(vals) for vals in s])
        sv = np.asarray([np.dot(s_, v_) for s_, v_ in zip(sing_matrix, v)])
        usv = np.asarray([np.dot(u_, sv_) for u_, sv_ in zip(u, sv)])
    else:
        sing_matrix = np.diag(s)
        usv = np.dot(u, np.dot(sing_matrix, v))
    return usv

def true_one(d):
    truthy = [(k,v) for k, v in d.items() if v]
    if len(truthy) != 1:
        return False
    return truthy[0]

def get_rank(rank_mode, rank_value, s, image_shape):
    if rank_mode == 'rank':
        if rank_value != int(rank_value) or rank_value <= 0:
            raise click.ClickException('rank must be an integer > 0')
        # reduce rank to max if use entered above max
        rank = min(rank_value, *image_shape)
    elif rank_mode == 'ratiorank':
        if not (0.0 < rank_value <= 1.0):
            raise click.ClickException('ratio must be in (0, 1]')
        n_sing_vals = s.shape[0]
        # ensure we have at least 1 singular value
        rank = max(int(n_sing_vals * rank_value), 1)
    else:
        rank = find_n_that_equalizes_values(image_shape)
    return rank

@click.command()
@click.argument('input', type=click.File('rb'))
@click.argument('output', type=click.File('wb'))
@click.option('--colorspace', '-c', default='RGB',
    type=click.Choice(colorspaces_by_name.keys()))
@click.option('--rank', '-r', 'rank_mode', type=click.INT)
@click.option('--ratiorank', '-f', 'ratiorank_mode', type=click.FLOAT)
@click.option('--equal', '-e', 'equal_mode', is_flag=True)
@click.option('--verbose', '-v', count=True)
def main(input, output, colorspace, rank_mode, ratiorank_mode, equal_mode,
         verbose):
    if verbose:
        logger.setLevel('INFO')

    colorspace_obj = colorspaces_by_name[colorspace]

    rank_modes = {'rank': rank_mode,
                  'ratiorank': ratiorank_mode,
                  'equal': equal_mode}
    selected_rank_mode = true_one(rank_modes)
    if not selected_rank_mode:
        raise click.ClickException('must select ONE of (--rank, --ratiorank, '
                                   '--equal)')

    logger.info('opening %s', input.name)
    image = Image.open(input)
    if image.mode != colorspace:
        logger.info('converting image to %s', colorspace)
        image = image.convert(mode=colorspace)

    image_shape = image.size
    image_band_names = image.getbands()

    logger.info('representing image as matrix')
    image_data = scipy.misc.fromimage(image)

    # generalize single-band images to images where the bands are the 3rd
    # dimension
    if colorspace == 'L':
        image_data = image_data[...,np.newaxis]

    bitmap_values = image_data.size
    svd_values = 0

    for band in range(image_data.shape[2]):
        logger.info('factoring, truncating, and multiplying band %s',
                    image_band_names[band])
        # full_matrices=False maybe provides better performance?
        # for an (M,N) matrix A, svd(A) produces:
        #   U.shape => (M,min(M,N))
        #   and
        #   V.shape => (min(M,N),N)
        #   (instead of (M,M) and (N,N))
        # this is good because the max rank an is min(M,N) anyway...so less
        # memory needed/faster?
        U, s, V = scipy.linalg.svd(image_data[...,band], full_matrices=False)
        rank = get_rank(*selected_rank_mode, s=s, image_shape=image_shape)
        logger.info("\told rank = %d, new rank = %d", len(s), rank)
        U, s, V = truncate_svd_factors(U, s, V, rank)
        S = scipy.linalg.diagsvd(s, s.shape[0], V.shape[0])
        image_data[...,band] = np.clip(U.dot(S.dot(V)), 0.0, 255.0)
        svd_values += U.size + s.size + V.size

    if colorspace == 'L':
        # put back in 2 dimensions
        image_data = image_data.reshape(image_data.shape[:2])
        channel_axis=1
    else:
        channel_axis=2

    new_image = scipy.misc.toimage(image_data, mode=colorspace,
                                   channel_axis=channel_axis)
    logger.info('saving image to %s', output.name)
    new_image.save(output)

    # stats printing
    logger.info('[stats] values in bitmap: %d', bitmap_values)
    logger.info('[stats] values in svd:    %d (%.2f%%)', svd_values, 100.0*float(svd_values)/bitmap_values)

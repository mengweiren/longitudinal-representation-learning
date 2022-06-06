import numpy as np
import os
from glob import glob

def normalize_img(array, percentile=None, zero_centered=True, verbose=False):
    min_ = np.min(array)
    max_ = np.percentile(array, percentile)
    #print(max_, min_)
    if verbose:
        print('original range: {},{}'.format( min_, max_))

    if max_ - min_ > 0:
        array = (array - min_)/ (max_ - min_)  # [0,1] normalized
    if zero_centered: # [-1,1] normalized
        array = array * 2 - 1
    if verbose:
        print('normalized to range {}, {}'.format(np.min(array), np.max(array)))
    return array


def renormalize_img(img, id_='', verbose=False):
    if verbose:
        print('{} | rescale from [{}.{}] to [0,1]'.format(id_, img.min(), img.max()))
    if img.max()- img.min() > 0:
        return (img-img.min())/(img.max()-img.min())
    else:
        return img

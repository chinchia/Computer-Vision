
import numpy as np
import cv2

def generate_wb_mask(img, pattern, fr, fb):
    '''
    Input:
        img: H*W numpy array, RAW image
        pattern: string, 4 different Bayer patterns (GRBG, RGGB, GBRG, BGGR)
        fr: float, white balance factor of red channel
        fb: float, white balance factor of blue channel 
    Output:
        mask: H*W numpy array, white balance mask
    '''
    ########################################################################
    # TODO:                                                                #
    #   1. Create a numpy array with shape of input RAW image.             #
    #   2. According to the given Bayer pattern, fill the fr into          #
    #      correspinding red channel position and fb into correspinding    #
    #      blue channel position. Fill 1 into green channel position       #
    #      otherwise.                                                      #
    ########################################################################

    mask = np.zeros(img.shape[:2])
    
    if pattern == 'GRBG':
        mask[::2, ::2] = 1
        mask[::2, 1::2] = fr
        mask[1::2, ::2] = fb
        mask[1::2, 1::2] = 1
        
    elif pattern == 'RGGB':
        mask[::2, ::2] = fr
        mask[::2, 1::2] = 1
        mask[1::2, ::2] = 1
        mask[1::2, 1::2] = fb
        
    elif pattern == 'GBRG':
        mask[::2, ::2] = 1
        mask[::2, 1::2] = fb
        mask[1::2, ::2] = fr
        mask[1::2, 1::2] = 1
    
    elif pattern == 'BGGR':
        mask[::2, ::2] = fb
        mask[::2, 1::2] = 1
        mask[1::2, ::2] = 1
        mask[1::2, 1::2] = fr

    ########################################################################
    #                                                                      #
    #                           End of your code                           #
    #                                                                      # 
    ########################################################################
        
    return mask
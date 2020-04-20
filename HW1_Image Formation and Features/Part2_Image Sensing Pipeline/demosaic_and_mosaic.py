
import numpy as np

from demosaic_2004 import demosaicing_CFA_Bayer_Malvar2004

def mosaic(img, pattern):
    '''
    Input:
        img: H*W*3 numpy array, input image.
        pattern: string, 4 different Bayer patterns (GRBG, RGGB, GBRG, BGGR)
    Output:
        output: H*W numpy array, output image after mosaic.
    '''
    ########################################################################
    # TODO:                                                                #
    #   1. Create the H*W output numpy array.                              #   
    #   2. Discard other two channels from input 3-channel image according #
    #      to given Bayer pattern.                                         #
    #                                                                      #
    #   e.g. If Bayer pattern now is BGGR, for the upper left pixel from   #
    #        each four-pixel square, we should discard R and G channel     #
    #        and keep B channel of input image.                            #     
    #        (since upper left pixel is B in BGGR bayer pattern)           #
    ########################################################################

    # create target array, twice the size of the original image
    output = np.zeros(img.shape[:2])
    
    if pattern == 'GRBG':
    # map the RGB values according to the GRBG pattern
        output[::2, ::2] = img[::2, ::2, 1]
        output[::2, 1::2] = img[::2, 1::2, 0]
        output[1::2, ::2] = img[1::2, ::2, 2]
        output[1::2, 1::2] = img[1::2, 1::2, 1]
        
    elif pattern == 'RGGB':
    # map the RGB values according to the RGGB pattern
        output[::2, ::2] = img[::2, ::2, 0]
        output[::2, 1::2] = img[::2, 1::2, 1]
        output[1::2, ::2] = img[1::2, ::2, 1]
        output[1::2, 1::2] = img[1::2, 1::2, 2]
        
    elif pattern == 'GBRG':
    # map the RGB values according to the GBRG pattern
        output[::2, ::2] = img[::2, ::2, 1]
        output[::2, 1::2] = img[::2, 1::2, 2]
        output[1::2, ::2] = img[1::2, ::2, 0]
        output[1::2, 1::2] = img[1::2, 1::2, 1]
    
    elif pattern == 'BGGR':
    # map the RGB values according to the BGGR pattern
        output[::2, ::2] = img[::2, ::2, 2]
        output[::2, 1::2] = img[::2, 1::2, 1]
        output[1::2, ::2] = img[1::2, ::2, 1]
        output[1::2, 1::2] = img[1::2, 1::2, 0]
    
    ########################################################################
    #                                                                      #
    #                           End of your code                           #
    #                                                                      # 
    ########################################################################

    return output


def demosaic(img, pattern):
    '''
    Input:
        img: H*W numpy array, input RAW image.
        pattern: string, 4 different Bayer patterns (GRBG, RGGB, GBRG, BGGR)
    Output:
        output: H*W*3 numpy array, output de-mosaic image.
    '''
    #### Using Python colour_demosaicing library
    #### You can write your own version, too
    output = demosaicing_CFA_Bayer_Malvar2004(img, pattern)
    output = np.clip(output, 0, 1)

    return output



import numpy as np
import math

def tone_mapping(img, I, B, index=0, inv=False):
    '''
    Input:
        img: H*W*3 numpy array, input image.
        I: 201*1024 array, represents 201 tone curves for Irradiance.
        B: 201*1024 array, represents 201 tone curves for Brightness.
        index: int, choose which curve to use, default is 0
        inv: bool, judge whether tone mapping (False) or inverse tone mapping (True), default is False
    Output:
        output: H*W*3 numpy array, output image afte (inverse) tone mapping.
    '''
    if inv == True:
    	output = np.interp(img.ravel(), B[index], I[index])
    elif inv == False:
    	output = np.interp(img.ravel(), I[index], B[index])
    output = output.reshape(img.shape)

    return output
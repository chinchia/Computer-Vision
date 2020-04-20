# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 16:36:28 2019

@author: g1022
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt

origin_img = cv2.imread('D:/course/computer vision/Homework1/hw1_1/original.jpg')
# convert the image into grayscale
img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY)

######################################
#####  Gaussian kernel function  #####
######################################
def gaussian_kernel(kernel_size, sigma):
    size = kernel_size // 2
    
    if kernel_size % 2 == 0:    # asymmetric (usually don't use this)
        x, y = np.mgrid[-size:size, -size:size]
    else:
        x, y = np.mgrid[-size:size+1, -size:size+1]
        
    gaussian_kernel = (np.exp((-(x**2 + y**2)) / (2 * sigma**2))) / (2 * np.pi * sigma**2)
    # normalization
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
    
    return gaussian_kernel

##################################
#####  convolution function  #####
##################################
def convolution(image, kernel):
    # zero padding
    img_row, img_col = image.shape
    kernel_row, kernel_col = kernel.shape
    pad_height = kernel_row // 2
    pad_width = kernel_col // 2
    pad_img = np.zeros((img_row + (2 * pad_height), img_col + (2 * pad_width)))
    pad_img[pad_height:pad_img.shape[0]-pad_height, pad_width:pad_img.shape[1]-pad_width] = image
    # convolution (kernel to image)
    blur_img = np.zeros(image.shape)
    for row in range(img_row):
        for col in range(img_col):
            blur_img[row, col] = np.sum(kernel * pad_img[row:row+kernel_row, col:col+kernel_col])
    
    return blur_img

######################################
#####  Gaussian smooth function  #####
######################################
# bigger sigma or bigger kernel size, the image becomes more blurred
def gaussian_smooth(image, sigma, kernel_size):
    kernel = gaussian_kernel(kernel_size=kernel_size, sigma=sigma)
    blur_img = convolution(image=image, kernel=kernel)
    
    return blur_img
    

blur_size5 = gaussian_smooth(image=img, sigma=5, kernel_size=5)
imgplot = plt.imshow(blur_size5)
plt.show()
blur_size10 = gaussian_smooth(image=img, sigma=5, kernel_size=10)
imgplot = plt.imshow(blur_size10)
plt.show()

####################################
#####  Sobel filters function  #####
####################################
def sobel_edge_detection(image):
    # create 3X3 kernel
    h_mask = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])
    v_mask = np.array([[1,2,1], [0,0,0], [-1,-2,-1]])
    # compute image gradient (horizontal and vertical)
    img_x = convolution(image=image, kernel=h_mask)
    img_y = convolution(image=image, kernel=v_mask)
    # eliminate weak gradients by proper threshold
    filtered_x =  np.absolute(img_x)
    filtered_x[filtered_x < 20] = 0
    filtered_y =  np.absolute(img_y)
    filtered_y[filtered_y < 20] = 0
    
    # compute the magnitude of gradient
    magnitude = np.sqrt(np.square(filtered_x) + np.square(filtered_y))

    return magnitude, filtered_x, filtered_y


sobel_size5 = sobel_edge_detection(image=blur_size5)
imgplot = plt.imshow(sobel_size5[0])
plt.show()
imgplot = plt.imshow(sobel_size5[1])
plt.show()
imgplot = plt.imshow(sobel_size5[2])
plt.show()

sobel_size10 = sobel_edge_detection(image=blur_size10)
imgplot = plt.imshow(sobel_size10[0])
plt.show()
imgplot = plt.imshow(sobel_size10[1])
plt.show()
imgplot = plt.imshow(sobel_size10[2])
plt.show()


#######################################
#####  structure tensor function  #####
#######################################
def structure_tensor(image, k, window_size):
    magnit, imx, imy = sobel_edge_detection(image)
    
    # compute the structure tensor component
    Ixx = imx ** 2
    Ixy = imx * imy
    Iyy = imy ** 2
    
    Ixx = gaussian_smooth(Ixx, sigma=5, kernel_size=window_size)
    Ixy = gaussian_smooth(Ixy, sigma=5, kernel_size=window_size)
    Iyy = gaussian_smooth(Iyy, sigma=5, kernel_size=window_size)
    
    # calculate the measurement of corner response (Harris detector)
    det = Ixx * Iyy - Ixy ** 2
    trace = Ixx + Iyy
    R = det - k * np.square(trace)
    
    # dilate the pixels, then filter out corners by specific threshold
    dilated = cv2.dilate(R, None)
    threshold = 0.0005 * dilated.max()
    corners = origin_img
    corners[dilated > threshold] = [255, 0, 0]
    
    return corners, R


corners_size3 = structure_tensor(image=sobel_size10[0], k=0.05, window_size=3)
imgplot = plt.imshow(corners_size3[0])
plt.show()
corners_size30 = structure_tensor(image=sobel_size10[0], k=0.05, window_size=30)
imgplot = plt.imshow(corners_size30[0])
plt.show()


##############################################
#####  non-maximum suppression function  #####
##############################################
def nms(R, sigma):
    # dilate the pixels, then filter out corners by specific threshold
    threshold = 0.0005 * R.max()
    radius = 2 * sigma
    Nx, Ny = R.shape
    corners = []
    skip = np.empty(R.shape)
    for i in range(0, R.shape[0]):
        for j in range(0, R.shape[1]):
            if R[i][j] < threshold:
                skip[i][j] = True
            else:
                skip[i][j] = False
    for i in range(radius, Ny-radius):
        j = radius
        while j < Nx - radius and (skip[i][j] or R[i][j-1] >= R[i][j]):
            j = j + 1
        while j < Nx - radius:
            while j < Nx - radius and (skip[i][j] or R[i][j+1] >= R[i][j]):
                j = j + 1
            if j < Nx - radius:
                p1 = j + 2
                while p1 <= j + radius and R[i][p1] < R[i][j]:
                    skip[i][p1] = True
                    p1 = p1 + 1
                if p1 > j + radius:
                    p2 = j - 1
                    while p2 >= j - radius and R[i][p2] <= R[i][j]:
                        p2 = p2 - 1
                    if p2 < j - radius:
                        k = i + radius
                        found = False
                        while not found and k > i:
                            l = j + radius
                            while not found and l >= j - radius:
                                if R[k][l] > R[i][j]:
                                    found = True
                                else:
                                    skip[k][l] = True
                                l = l - 1
                            k = k - 1
                        k = i - radius
                        while not found and k < i:
                            l = j - radius
                            while not found and l <= j + radius:
                                if R[k][l] >= R[i][j]:
                                    found = True
                                l = l + 1
                            k = k + 1
                        if not found:
                            corners.append([i, j, R[i][j]])
                j = p1
    
    
    return corners
    

origin_img = cv2.imread('D:/course/computer vision/Homework1/hw1_1/original.jpg')

nms_corner = nms(corners_size3[1], sigma=5)
nms_corner_size3 = origin_img
for i in nms_corner:
    nms_corner_size3[i[0]][i[1]] = [255, 0, 0]
imgplot = plt.imshow(nms_corner_size3)
plt.show()

nms_corner = nms(corners_size30[1], sigma=5)
nms_corner_size30 = origin_img
for i in nms_corner:
    nms_corner_size30[i[0]][i[1]] = [255, 0, 0]
imgplot = plt.imshow(nms_corner_size30)
plt.show()


##########################
#####  rotate image  #####
##########################
def rotate(image, angle, center=None, scale=1.0):
    row, col = image.shape[:2]
    # set the center of image to be the rotate center if the rotate center is not assigned
    if center is None:
        center = (col / 2, row / 2)
 
    # do the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (row, col))

    return rotated


rotated_img = rotate(nms_corner_size3, 30)
imgplot = plt.imshow(rotated_img)
plt.show()

    
##########################
#####  resize image  #####
##########################
def scaled(image, factor):
    width = int(image.shape[1] * factor)
    height = int(image.shape[0] * factor)
    dim = (width, height)
    resized = cv2.resize(image, dim)
    
    return resized

scaled_img = scaled(nms_corner_size3, 0.5)
imgplot = plt.imshow(scaled_img)
plt.show()





import math
import sys
import numpy
import scipy.ndimage as nd

"""
Function to perform Laplacian of Gaussian unsharp masking
Input:
       Im: Gray-scale (uint8)
    sigma: Standard deviation (positive), default=1.0
        R: Radius (positive), default=3
Output: 
       Ie: Gray-scale (uint8)

Description 
enh_sharpLUM performs unsharp masking using the Laplacian of Gaussian.
Given a radius of 3, and a standard deviation of 1.0, the algorithm
uses a kernel of the form:

     0.0003    0.0026    0.0086    0.0124    0.0086    0.0026    0.0003
     0.0026    0.0175    0.0392    0.0431    0.0392    0.0175    0.0026
     0.0086    0.0392         0   -0.0965         0    0.0392    0.0086
     0.0124    0.0431   -0.0965   -0.3183   -0.0965    0.0431    0.0124
     0.0086    0.0392         0   -0.0965         0    0.0392    0.0086
     0.0026    0.0175    0.0392    0.0431    0.0392    0.0175    0.0026
     0.0003    0.0026    0.0086    0.0124    0.0086    0.0026    0.0003

Ref(s):
Schreiber, W.F., "Wirephoto quality improvement by unsharp masking",
Pattern Recognition, 1970, Vol.2, No.2, pp.117-120. 

Gonzalez, R.C., Woods, R.E., "Digital Image Processing (2nd ed.)",
Prentice Hall, 2002.
"""
def enh_sharpLUM(img, sigma=1.0, R=3):
	
	# Derive the Laplacian of Gaussian masking kernel (positive discrete Laplacian)
	D = R * 2 + 1 # define the kernel diameter
	kernel = numpy.zeros([D,D])
	
	for i in range(-R,R+1):
	    for j in range(-R,R+1):
	        ki = i + R
	        kj = j + R
	        X = ((i**2+j**2)/(2*sigma**2));
	        kernel[ki,kj] = -(1.0/(numpy.pi*sigma**4.0)) * (1.0-X) * numpy.exp(-X)
		
	# Apply the unsharp masking kernel
	imgK = nd.convolve(img, kernel, mode='constant', cval=0.0)

	# Subtract the kernel from the original image
	imgS = img - imgK
	
	return imgS

"""
Function to perform Nagao-Matsuyama edge-preserving smoothing
"""
def nagao_matsuyama(im,n=5):

    #Create an empty image to store the value into.
    img = numpy.zeros(im.shape,dtype=numpy.int16)
    
    # Store the indices for each of the 9 regions
    A = [6,7,8,11,12,13,16,17,18]
    B = [1,2,3,6,7,8,12]
    C = [8,9,12,13,14,18,19]
    D = [12,16,17,18,21,22,23]
    E = [5,6,10,11,12,15,16]
    F = [0,1,5,6,7,11,12]
    G = [3,4,7,8,9,12,13]
    H = [12,13,17,18,19,23,24]
    I = [11,12,15,16,17,20,21]

    regions = [A,B,C,D,E,F,G,H,I]
    blockMedians = range(9)
    blockVariance = range(9)
    
    v = (n-1) / 2

    #Start Processing the image
    for i in range(2,im.shape[0]-2):
        for j in range(2,im.shape[1]-2):
            # Extract the neighbourhood and the pixel from the image
            block = im[i-v:i+v+1, j-v:j+v+1]
            
            # Convert to a vector
            vBlock = block.flatten()
            
            # Calculate the means and variances
            for k in range(0, len(regions)):
                blockMedians[k] = numpy.mean(numpy.take(vBlock,regions[k]))
                blockVariance[k] = numpy.var(numpy.take(vBlock,regions[k]))
            
            # Calculate the smallest variance
            minVar = min(blockVariance)

            #Set the pixel to the value of the mean that was associated with the variance
            img[i,j] = blockMedians[blockVariance.index(minVar)]

        print i, j
    return img





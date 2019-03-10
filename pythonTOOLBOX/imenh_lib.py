# Library to perform image enhancement:
#
# enh_alphaTMean    - noise reduction using alpha-trimmed mean filtering
# enh_truncMedian   - noise reduction using truncated median "mode" filtering
# enh_hybridMedian  - noise reduction using hybrid median filtering


import numpy
import math
import scipy.ndimage as nd
import pylab

# Function to perform noise reduction using alpha-trimmed means
#
# alpha is a parameter describing how many elements to trim from the
# sorted vector representing the image neighborhood. For example in a
# 5x5 neighborhood, with alpha=0.2, 5 elements would be trimmed from either end.
# 
# For example:
# 
#        23 16 23 43 54
#        12 23 34 54 32
#        23 34 49 3  21
#        24 55 65 12 4
#         8  9  7 21 12
#         
# Vector:
#      
#     (3,4,7,8,9,12,12,12,16,21,21,23,23,23,23,24,32,34,34,43,49,54,54,55,65) 
#      
# Mean = 26
#      
# Trimmed Vector (alpha = 0.2 elements):
#      
#     (12,12,12,16,21,21,23,23,23,23,24,32,34,34,43) 
#      
# Trimmed Mean = 23
#         
# The value of alpha is restricted to the interval 0<=alpha<0.5 Ideally in a
# 5x5 neighbourhood, the use of alpha=0.5 will result in the median filter
#
# Ref(s):
#   Bednar, J.B., Watt, T.L., "Alpha-trimmed means and their relationship
#   to median filters", IEEE Transactions on Acoustics, Speech and Signal
#   Processing, Vol.32(1), pp.145-153 (1984)
#
def enh_alphaTMean(im,alpha,n=5):
    img = numpy.zeros(im.shape,dtype=numpy.int16)
    
    v = (n-1)/2
    
    # Calculate the trim coefficient
    b = int((n*n)*(alpha))
    
	# Process the image
    for i in range(0,im.shape[0]):
        for j in range(0,im.shape[1]):
            # Extract the window area
            block = im[max(i-v,0):min(i+v+1,im.shape[0]), max(j-v,0):min(j+v+1,im.shape[1])]

            # Reshape the neighborhood into a vector by flattening the 2D block
            wB = block.flatten()
            
            # Sort the vector into ascending order
            wB = numpy.sort(wB)
            len = wB.size
            
            # Trim b elements from each end of the vector
            if (b != 0):
                nwB = wB[b:len-b]
    
            # Calculate the mean of the trimmed vector
            tMean = nwB.mean()

            # Assign the values
            if (tMean > 0):
                img[i][j] = int(tMean)
    return img
    
# Function to perform noise reduction using truncated median "mode"
#
# Ref(s):
#   Davies, E.R., "On the noise suppression and image enhancement
#   characteristics of the median, truncated median and mode filters", 
#   Pattern Recognition Letters, Vol.7, pp.87-97 (1988)
#
def enh_truncMedian(im,n=5):

    img = numpy.zeros(im.shape,dtype=numpy.int16)
    
    v = (n-1) / 2

	# Process the image
    for i in range(0,im.shape[0]):
        for j in range(0,im.shape[1]):
            # Extract the window area
            block = im[max(i-v,0):min(i+v+1,im.shape[0]), max(j-v,0):min(j+v+1,im.shape[1])]
            print block.shape
            # Reshape the neighborhood into a vector by flattening the 2D block
            wB = block.flatten()

            # Calculate vector statistics
            wMean = wB.mean()
            wMin = wB.min()
            wMax = wB.max()
            wMed = numpy.median(wB)
            
            # Calculate the bounds, and select the appropriate elements
            if (wMed < wMean):
                upper = 2 * wMed - wMin
                NwB = wB.compress((wB<upper).flat)
            else:
                lower = 2 * wMed - wMax
                NwB = wB.compress((wB>lower).flat)
           
            # Calculate the median of the selected elements
            xmed = numpy.median(NwB)
            
            # Assign the values               
            if (xmed > 0):
                img[i][j] = int(xmed)
            else:
                img[i][j] = im[i][j]
    return img
    
# Function to perform noise reduction using hybrid median
# 
# Extracts medians from a 5x5 neighbourhood using two masks: 
# 
#        1 0 0 0 1          0 0 1 0 0
#        0 1 0 1 0          0 0 1 0 0
#        0 0 1 0 0   and    1 1 1 1 1
#        0 1 0 1 0          0 0 1 0 0
#        1 0 0 0 1          0 0 1 0 0
#
# The hybrid median is then calculated as the median of three values comprising
# the two medians above and the pixel in the original image.
#
# Ref(s):
#
def enh_hybridMedian(im,n=5):

    img = numpy.zeros(im.shape,dtype=numpy.int16)
    
    # Derive indices for the two patterns representing X and +
    indicesC = [0,4,6,8,12,16,18,20,24]
    indicesP = [2,7,10,11,12,13,14,17,22]
    
    v = (n-1) / 2

	# Process the image (ignoring the outer two layers of the image boundary
    for i in range(2,im.shape[0]-2):
        for j in range(2,im.shape[1]-2):
            # Extract the neighbourhood area
            block = im[i-v:i+v+1, j-v:j+v+1]
            
            # Reshape the neighborhood into a vector by flattening the 2D block
            wB = block.flatten()
            
            # Extract pixel values using indices
            wBc = numpy.take(wB,indicesC)
            wBp = numpy.take(wB,indicesP)
                  
            # Calculate the median values      
            wBcMed = numpy.median(wBc)
            wBpMed = numpy.median(wBp)
            
            # Calculate the hybrid median of the original pixel, and the two 
            # medians extracted above
            xmed = numpy.median([wBcMed,wBpMed,im[i][j]])

            # Assign the values               
            if (xmed > 0):
                img[i][j] = int(xmed)
            else:
                img[i][j] = im[i][j]
    return img

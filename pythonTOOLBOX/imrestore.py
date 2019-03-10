# Library to perform colour restoration via white balancing:
#
# grayWorld  - Performs white-balancing through grayworld algorithm
# SDWGW      - Performs white-balancing through standard deviation with grayworld
# peakdetect - Detects peaks in a histogram
# stretch    - Stretches each histogram in R,G,B components

import numpy
from numpy import NaN, Inf
import math
import scipy.ndimage as nd
import pylab

# Function to perform grayworld white-balancing
# Scales R and B channels so they have the same mean as the G channel
# For a typical scene, the average intensity of the R,G,B channels
# should be equal.
#
def grayWorld(imgRGB):
    img = numpy.zeros(imgRGB.shape)

    imR = imgRGB[:,:,0]
    imG = imgRGB[:,:,1]
    imB = imgRGB[:,:,2]
  
    # Calculate the mean of each colour channel
    meanR = imR.mean();
    meanG = imG.mean();
    meanB = imB.mean();
    
    # Calculate transformation coefficients (green remains unchanged)
    kr = meanG / numpy.double(meanR)
    kb = meanG / numpy.double(meanB)
    	
	# Apply grayworld transformation - adjust red and blue pixels
    img[:,:,0] = numpy.minimum((kr * imR),255)
    img[:,:,1] = imG
    img[:,:,2] = numpy.minimum((kb * imB),255)
        
    return img.astype('uint8')


# Function to perform max-white white balancing algorithm
#
def maxWhite(imgRGB):
    img = numpy.zeros(imgRGB.shape)

    imR = imgRGB[:,:,0]
    imG = imgRGB[:,:,1]
    imB = imgRGB[:,:,2]
    
    # Calculate the maximum of each colour channel
    maxR = imR.max();
    maxG = imG.max();
    maxB = imB.max();

    # Calculate transformation coefficients
    kr = 255.0 / numpy.double(maxR)
    kg = 255.0 / numpy.double(maxG)
    kb = 255.0 / numpy.double(maxB)
    	    	
	# Apply max-White transformation
    img[:,:,0] = numpy.minimum((kr * imR),255)
    img[:,:,1] = numpy.minimum((kg * imG),255)
    img[:,:,2] = numpy.minimum((kb * imB),255)
            
    return img.astype('uint8')


# Lam, H.-K. Au, O., Wong, C.-W., "Automatic white balancing using 
# standard deviation of rgb components", Proc. of the Int. Conf. 
# on Circuits and Systems, pp.921-924 (2004)
#
def SDWGW(imgRGB,nBlocks=20):
    img = numpy.zeros(imgRGB.shape)
    
    dx = imgRGB.shape[0] / nBlocks 
    dy = imgRGB.shape[1] / nBlocks 
    
    imR = imgRGB[:,:,0]
    imG = imgRGB[:,:,1]
    imB = imgRGB[:,:,2]

    N = dx * dy
    
    imgSDr = numpy.zeros(N)
    imgMNr = numpy.zeros(N) 
    imgSDg = numpy.zeros(N)
    imgMNg = numpy.zeros(N) 
    imgSDb = numpy.zeros(N)
    imgMNb = numpy.zeros(N) 
    
    t = 0   
    # Extract the blocks and calculate the mean and standard deviation for each
    # of the colour components. Each is stored in a vector.
    for i in xrange(dx):
        for j in xrange(dy):
            blockR = imgRGB[i*nBlocks:i*nBlocks+nBlocks,j*nBlocks:j*nBlocks+nBlocks,0]
            blockG = imgRGB[i*nBlocks:i*nBlocks+nBlocks,j*nBlocks:j*nBlocks+nBlocks,1]
            blockB = imgRGB[i*nBlocks:i*nBlocks+nBlocks,j*nBlocks:j*nBlocks+nBlocks,2]
            imgSDr[t] = blockR.std()
            imgMNr[t] = blockR.mean()
            imgSDg[t] = blockG.std()
            imgMNg[t] = blockG.mean()
            imgSDb[t] = blockB.std()
            imgMNb[t] = blockB.mean()
            t = t + 1
            
    # Calculate the sum for each of the SD vectors        
    imgSDrSUM = imgSDr.sum()
    imgSDgSUM = imgSDg.sum()
    imgSDbSUM = imgSDb.sum()
          
    SDWAr = 0.0
    SDWAg = 0.0
    SDWAb = 0.0

    # Calculate SD weighted average (SDWA) of each colour channel (Eq.4)
    for k in xrange(N):
        SDWAr = SDWAr + (imgSDr[k]/imgSDrSUM) * imgMNr[k]
        SDWAg = SDWAg + (imgSDg[k]/imgSDgSUM) * imgMNg[k]   
        SDWAb = SDWAb + (imgSDb[k]/imgSDbSUM) * imgMNb[k]
        
    # Calculate transformation coefficients (Eq.5)
    SDWAavg = (SDWAr + SDWAg + SDWAb) / 3.0
    Rgain = SDWAavg / SDWAr
    Ggain = SDWAavg / SDWAg
    Bgain = SDWAavg / SDWAb
        
    # Apply SDWGW transformation
    img[:,:,0] = numpy.minimum((Rgain * imR),255)
    img[:,:,1] = numpy.minimum((Ggain * imG),255)
    img[:,:,2] = numpy.minimum((Bgain * imB),255)
    
    print img[:,:,0].min(), img[:,:,0].max()

    return img.astype('uint8')

# Detects peaks in a histogram
# Finds the local maxima and minima ("peaks") in the vector v.
# A point is considered a maximum peak if it has the maximal
# value, and was preceded (to the left) by a value lower by delta.
#
# If x exists, the indices in MAXTAB and MINTAB are replaced with 
# the corresponding X-values.
# Eli Billauer, 3.4.05 (Explicitly not copyrighted).
#
def peakdetect(v, delta, x=None):

    maxtab = []
    mintab = []
       
    if x is None:
        x = numpy.arange(len(v))
    
    v = numpy.asarray(v)
    
    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')
    
    if not numpy.isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    
    if delta <= 0:
        sys.exit('Input argument delta must be positive')
    
    mn, mx = Inf, -Inf
    mnpos, mxpos = NaN, NaN
    
    lookformax = True
    
    for i in numpy.arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        
        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True
 
    return numpy.array(maxtab), numpy.array(mintab)

# Function to stretch each of the colour components
#
def stretch(imgRGB):
    img = numpy.zeros(imgRGB.shape)

    imR = imgRGB[:,:,0]
    imG = imgRGB[:,:,1]
    imB = imgRGB[:,:,2]
    
    # Calculate the minimum of each colour channel
    #minR = imR.min();
    #minG = imG.min();
    #minB = imB.min();

    # Subtract the minimum
    #img[:,:,0] = numpy.maximum(0,(imR-minR))
    #img[:,:,1] = numpy.maximum(0,(imG-minG))
    #img[:,:,2] = numpy.maximum(0,(imB-minB))

    #imgW = maxWhite(img)
    
    # Get the red image histogram
    hstR,bins = numpy.histogram(imR.flatten(),256,(0,255),density=False)

    # Calculate the maximum peak in the histogram
    #p = numpy.unravel_index(hst.argmax(),hst.shape)
    maxP, minP = peakdetect(hstR, 1000)
    low = numpy.double(maxP[1][0])
    high = numpy.double(maxP[-1][0])
    print low, high
    
    for i in xrange(imgRGB.shape[0]):
        for j in xrange(imgRGB.shape[1]):
            if (imR[i][j] < low):
                img[i][j][0] = 0
            elif (imR[i][j] >= low and imR[i][j] <= high):
                img[i][j][0] = 255 * ((imR[i][j] - low)/(high - low))
            else: 
                img[i][j][0] = 255
 
     # Get the green image histogram
    hstG,bins = numpy.histogram(imG.flatten(),256,(0,255),density=False)

    # Calculate the maximum peak in the histogram
    #p = numpy.unravel_index(hst.argmax(),hst.shape)
    maxP, minP = peakdetect(hstG, 1000)
    low = numpy.double(maxP[1][0])
    high = numpy.double(maxP[-1][0])
    print low, high
   
    for i in xrange(imgRGB.shape[0]):
        for j in xrange(imgRGB.shape[1]):
            if (imG[i][j] < low):
                img[i][j][1] = 0
            elif (imG[i][j] >= low and imG[i][j] <= high):
                img[i][j][1] = 255 * ((imG[i][j] - low)/(high - low))
            else: 
                img[i][j][1] = 255

     # Get the green image histogram
    hstB,bins = numpy.histogram(imB.flatten(),256,(0,255),density=False)

    # Calculate the maximum peak in the histogram
    #p = numpy.unravel_index(hst.argmax(),hst.shape)
    maxP, minP = peakdetect(hstB, 1000)
    low = numpy.double(maxP[1][0])
    high = numpy.double(maxP[-1][0])
    
    for i in xrange(imgRGB.shape[0]):
        for j in xrange(imgRGB.shape[1]):
            if (imB[i][j] < low):
                img[i][j][2] = 0
            elif (imB[i][j] >= low and imB[i][j] <= high):
                img[i][j][2] = 255 * ((imB[i][j] - low)/(high - low))
            else: 
                img[i][j][2] = 255
    print low, high
 
    return img.astype('uint8')

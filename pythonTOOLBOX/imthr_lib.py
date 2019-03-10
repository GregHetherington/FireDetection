# Library to perform image thresholding:
#
# ADAPTIVE
#  niblack    - Niblack's algorithm 
#  bernsen    - Bernsen's algorithm (DRAFT CODE)
#  sauvola    - Sauvola's algorithm
#
# CLUSTERING
#  otsu       - Otsu's algorithm
#  minError   - Minimum Error algorithm (Kittler)
#
# ATTRIBUTE
#  moments    - Moment-preserving (Tsai)
#
# ENTROPY
#  maximumE   - Maximum entropy, Kapur's algorithm
#
# MISC FUNCTIONS
#  im2bw      - Function to perform global thresholding

import numpy
import math
import scipy.ndimage as nd
import pylab

# Function to perform maximum entropy based thresholding using Kapurs's algorithm
#
# Parameters:
# (in)    im     :  gray-scale image
# (out)   tvalue :  threshold value
#
# Ref(s):
# Kapur, J.N., Sahoo, P.K., Wong, A.K.C., "A new method for grey-level picture
# thresholding using the entropy of the histogram", Graphical Models and Image
# Processing, Vol.29(3), pp.273-285 (1985)
#
def maximumE(im):
    img = numpy.zeros(im.shape,dtype=numpy.int16)
    F = numpy.zeros(256)
   
    # Get the image histogram
    hst,bins = numpy.histogram(im.flatten(),256,(0,255),density=False)

    # Normalize the histogram 0->1
    pD = hst / numpy.float32(im.size)

    # Calculate the cumulative distribution function of the normalized histogram
    cdf = pD.cumsum() 
        
	# Calculate the function to be maximized
    for s in range(0,256):
        Ha = 0.0
        Hb = 0.0
        for i in range(0,s+1):
            if ((pD[i] > 0.0) and (cdf[i] > 0.0)):
                Ha = Ha + (-pD[i]/cdf[s] * flog(pD[i]/cdf[s]))  # (Eq.16)
                
        for j in range(s+1,256):
            if ((pD[j] > 0.0) and (cdf[j] > 0.0)):
                Hb = Hb + (-pD[j]/(1.0-cdf[s]) * flog(pD[j]/(1.0-cdf[s])))  # (Eq.17)
        F[s] = Ha + Hb
        
    # Calculate the maximum value amongst all the entropies
    tvalue = numpy.argmax(F)
    
    return tvalue


# Function to perform adaptive image thresholding using Sauvola's algorithm
#
# A modified version of Niblack's algorithm in which a threshold is computed
# with the dynamic range of standard deviation, R. The local mean is used to 
# multiply terms R and a fixed value k. This has the effect of amplifying the
# contribution of the SD in an adaptive manner.
#
# Parameters:
# (in)  im  :  gray-scale image
# (in)   R  :  dynamic range of standard deviation (default=128)
# (in)   k  :  parameter with positive values (default=0.5)
# (in)   n  :  size of the neighbourhood (nxn) (default=5)
# (out) img :  binary image
#
# Ref(s):
# Sauvola, J., Pietikainen, M., "Adaptive document image binarization", 
# Pattern Recognition, Vol.33, pp.225-236 (2000)
#
def sauvola(im,n=5,R=128,k=0.5):
    img = numpy.zeros(im.shape,dtype=numpy.int16)
    
    v = (n-1)/2
        
	# Process the image
    for i in range(0,im.shape[0]):
        for j in range(0,im.shape[1]):
            # Extract the neighbourhood area
            block = im[max(i-v,0):min(i+v+1,im.shape[0]), \
                       max(j-v,0):min(j+v+1,im.shape[1])]
            # Calculate the mean and standard deviation of the neighbourhood region
            wBmn = block.mean()
            wBstd = numpy.std(block)
            
            # Calculate the threshold value (Eq.5)
            wBTH = int(wBmn * (1 + k * ((wBstd/R) - 1)))
            
            # Threshold the pixel
            if (im[i][j] < wBTH):
                img[i][j] = 0
            else:
                img[i][j] = 255
            
    return img

# Function to perform adaptive image thresholding using Bernsen's algorithm
#
# Parameters:
# (in)  im  :  gray-scale image
# (in)  TH  :  Integer representing the local contrast threshold
# (in)  wS  :  Integer representing window size
# (in)  L   :  Integer representing (0) low and (1) high homogeneous areas
# (out) img :  binary image
#
# Ref(s):
# Bernsen, J., "Dynamic thresholding of grey-level images",
# Int. Conf. on Pattern Recognition, pp.1251-1255 (1986).
#
def bernsen(im,TH,wS,L):
    img = numpy.zeros(im.shape,dtype=numpy.int16)

    # Get the image histogram
    hst,bins = numpy.histogram(im.flatten(),nbr_bins,(0,255),density=False)

    maxR = math.floor(im.shape[0]/wS)
    maxC = math.floor(im.shape[1]/wS)
    
    bL = numpy.zeros([w,w])

	# Process the image
    for i in range(0,im.shape[0],maxR):
        bL[:,:] = 0
        for j in range(0,im.shape[1],maxC):
            # Extract the neighbourhood area
            block = im[i:i+(w-1), j:j+(w-1)]
            # Calculate the min and max of the neighbourhood region
            Zlow = block.min()
            Zhigh = block.max()
            
            if ((Zhigh - Zlow) < TH and (L == 1)):
                bL[:,:] = 255
            elif ((Zhigh - Zlow) < TH and (L == 0)):
                bL[:,:] = 0
            else:
                T = math.trunc((Zlow + Zhigh)/2.0)
                for x in range(1,w+1):
                    for y in range(1,w+1):
                        if (nI[x][y] >= T):
                            bL[x][y] = 255
            img[i:i+(w-1), j:j+(w-1)] = bL

    return img


# Function to perform image thresholding using Moment preservation
#
# Parameters:
# (in)  im     :  gray-scale image
# (out) tvalue :  threshold value
#
# Ref(s):
# Tsai, H., "Moments preserving thresholding: A new approach", 
# Computer Vision, Graphics and Image Processing, Vol.29(3), pp.377-393 (1985)
#
def moments(im):
    img = numpy.zeros(im.shape,dtype=numpy.int16)

    # Calculate the image histogram
    hst,bins = numpy.histogram(im.flatten(),256,(0,255),density=False)

    # Normalize the histogram 0->1
    hstpdf = hst / numpy.float32(im.size)

    m1 = 0.0
    m2 = 0.0
    m3 = 0.0
    
    # 0th moment
    m0 = 1.0
    
    # Calculate the moments 1-> 3 (Eq.2)
    for i in range(0,256):
        m1 = m1 + i * hstpdf[i]
        m2 = m2 + (i**2.0 * hstpdf[i])
        m3 = m3 + (i**3.0 * hstpdf[i])

    # Eqns from Appendix A1(i)
    cD = m0 * m2 - m1**2.0
    c0 = (m1 * m3 - m2**2.0) * (1.0/cD)
    c1 = (m1 * m2 - m0 * m3) * (1.0/cD)
    
    # Eqns from Appendix A1(ii)
    z0 = 0.5 * (-c1 - math.sqrt(c1**2.0 - 4*c0))
    z1 = 0.5 * (-c1 + math.sqrt(c1**2.0 - 4*c0))

    # Eqns from Appendix A1(iii)
    pd = z1 - z0
    p0 = (1.0/pd) * (z1 - m1)
    p1 = 1.0 - p0;

    # Choose the value of the threshold nearest to p0 with respect to 
    # the cumulative probabilities
    cdf = hstpdf.cumsum() 

    moments = cdf - p0
    moments = numpy.absolute(moments)
    
    tvalue = numpy.argmin(moments)
    
    return tvalue


# Function to perform adaptive image thresholding using Niblack's algorithm
#
# The threshold value at each pixel im[i][j] is calculated as the sum of 
# the mean and the standard deviation (times a constant, k) of the  
# neighbourhood surrounding the pixel (n = size of the neighbourhood).
# Note: Does not work well in images where the background has light texture
#
# Parameters:
# (in)  im  :  gray-scale image
# (in)   k  :  constant value
# (in)   n  :  size of the neighbourhood (nxn) (default=5)
# (out) img :  binary image
#
# Ref(s):
#   Niblack, W., An Introduction to Digital Image Processing, Prentice Hall
#   pp.115-116 (1986)
#
def niblack(im,k,n=5):
    img = numpy.zeros(im.shape,dtype=numpy.int16)
    
    v = (n-1)/2
        
	# Process the image
    for i in range(0,im.shape[0]):
        for j in range(0,im.shape[1]):
            # Extract the neighbourhood area
            block = im[max(i-v,0):min(i+v+1,im.shape[0]), \
                       max(j-v,0):min(j+v+1,im.shape[1])]
            # Calculate the mean and standard deviation of the neighbourhood region
            wBmn = block.mean()
            wBstd = numpy.std(block)
            
            # Calculate the threshold value
            wBTH = int(wBmn + k * wBstd)
            
            # Threshold the pixel
            if (im[i][j] < wBTH):
                img[i][j] = 0
            else:
                img[i][j] = 255
            
    return img

# Function to perform image thresholding using Otsu's algorithm
#
# Parameters:
# (in)  im     :  gray-scale image
# (out) tvalue :  threshold value
#
# Ref(s):
#   Otsu, N., "Threshold selection using grey level histograms", IEEE
#   Transactions on Systems, Man and Cybernetics, Vol.9(1), pp.62-66 (1979)
#
def otsu(im,nbr_bins=256):
    
    # Get the image histogram
    hst,bins = numpy.histogram(im.flatten(),nbr_bins,(0,255),density=False)

    # Normalize the histogram 0->1
    hstpdf = hst / numpy.float32(im.size)
    
    # Calculate the global mean (Eq.8)
    mu_T = 0.0
    for i in range(0,256):
        mu_T = mu_T + (i+1) * hstpdf[i]

    n = numpy.zeros(256)
    
    for t in range(0,256):
        
        # Calculate the zero- and first-order cumulative moments up to the t-th level
        w_t = numpy.sum(hstpdf[0:t+1])      # (Eq.6)
        
        mu_t = 0.0;
        for i in range(0,(t+1)):            # (Eq.7)
            mu_t = mu_t + (i+1) * hstpdf[i]
          
        # Calculate  the variance of the class separability (Eq.18)
        if ((w_t != 0.0) and (w_t != 1.0)):
            sigma = ((mu_T * w_t - mu_t)**2.0) / (w_t * (1.0-w_t))
        else:
            sigma = 0.0

        n[t] = sigma
        
    # Choose the threshold for which the variance of class separability is
    # at its maximum. Return the index of this value (Eq.19)
    tvalue = numpy.argmax(n)
    
    return tvalue
    
# Function to perform image thresholding using the Minimum error algorithm
#
# Parameters:
# (in)  im     :  gray-scale image
# (out) tvalue :  threshold value
#
# Ref(s):
#   Kittler, J., Illingworth, J., "Minimum error thresholding", Pattern
#   Recognition, Vol.19(1), pp.41-47 (1986)
#
def minError(im,nbr_bins=256):
    
    # Get the image histogram
    hst,bins = numpy.histogram(im.flatten(),nbr_bins,(0,255),density=False)

    F = numpy.zeros(256)
    tbest = 0
    
    # Compute the priori probability
    for t in range(0,256):

        # (Eq. 5,8,9 for i=1)
        a = numpy.sum(hst[0:t+1]) 
        aT = numpy.arange(1,t+2)

        b = 0.0
        if (a <= 0):
            b = 0.0
        else:
            u = numpy.sum(hst[0:t+1] * aT)
            u = u / a
            
            for k in range(0,t+1):   # (Eq.7 for i=1)
                x = (k-u)**2.0
                b = b + x * hst[k]
            
            b = b / a

        # (Eq. 5,8,9 for i=2)
        c = numpy.sum(hst[t+1:256]) 
        cT = numpy.arange(t+1,256)

        d = 0.0
        if (c <= 0):
            d = 0.0
        else:
            u = numpy.sum(hst[t+1:256] * cT)
            u = u / c
            
            for k in range(t+1,256):   # (Eq.7 for i=2)
                x = (k-u)**2.0
                d = d + x * hst[k]
            
            d = d / c
            
        # (Eq.15)
        F[t] = 1.0 + 2.0*(a*flog(b) + c*flog(d)) - 2.0*(a*flog(a) + c*flog(c))
                   
        if (F[t] < F[tbest]):
            tbest = t
            
    tvalue = tbest              
    
    return tvalue    
    
    
# Function to perform global image thresholding
#
# Parameters:
# (in)  im  :  gray-scale image
# (in)  thr :  threshold value (0->255)
# (out) img :  binary image
#
def im2bw(im,thr):
    img = numpy.zeros(im.shape,dtype=numpy.int16)

    for i in range(0,im.shape[0]):
        for j in range(0,im.shape[1]):
            if (im[i][j] < thr):
                img[i][j] = 0
            else:
                img[i][j] = 255
    return img    

# Function to trap errors in log10
#
def flog(x):
    if (x > 0):
        fL = math.log10(x)
    else:
        fL = 0.0
    return fL
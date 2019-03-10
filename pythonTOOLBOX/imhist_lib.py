# Library to process images using histogram modification algorithms:
#
# cumulativeHist - Create a cumulative histogram
# histeq         - Histogram equalization
# histmatch      - Histogram matching or specification
# histeqADAPT    - Adaptive histogram equalization
# histeqBI       - Bi-histogram equalization
# histhyper      - Histogram hyperbolization

import numpy
import math
import scipy.ndimage as nd
import pylab

# Function to derive a cumulative histogram
def cumulativeHist(hist, slopeMax=0):
    d = numpy.zeros((hist.size))
    c = 0
    for i in xrange(hist.size):
        c = c + hist[i]
        d[i] = c
    d = d / c
    if (slopeMax > 0):
        dh = 0
        for i in xrange(d.size-1):
            dh = dh + max(d[i+1]-dh-slopeMax-d[i],0)
            d[i+1] = d[i+1] - dh
    return d

# Function to perform contrast enhancement using image histogram equalization
def histeq(im,nbr_bins=256):
    img = numpy.zeros(im.shape,dtype=numpy.int16)

    # Calculate the histogram
    imhist,bins = numpy.histogram(im.flatten(),nbr_bins,(0,255),density=False)

    # Calculate the probability density function of the sub-histogram
    histpdf = imhist / numpy.float32(im.size)
    # Derive the cumulative histogram
    cdf = histpdf.cumsum()

    # Histogram equalization for each of the sub-histograms
    fL = 255 * cdf
    fL = fL.astype('uint8')
    print fL

    for i in range(0,im.shape[0]):
        for j in range(0,im.shape[1]):
            img[i][j] = fL[im[i][j]]

    xr = numpy.arange(0,256,1)
    pylab.plot(xr,fL)
    pylab.show()

    return img, imhist

# Function to perform image histogram specification or matching
#   Ref(s):
#   Gonzalez, R.C., Woods, R.E., "Digital Image Processing", 2nd ed.,
#   Prentice Hall, pp.94-102 (2001)

def histmatch(im1,im2,nbr_bins=256):
    # get image histograms
    hst1,bins1 = numpy.histogram(im1.flatten(),nbr_bins,(0,255),density=False)
    hst2,bins2 = numpy.histogram(im2.flatten(),nbr_bins,(0,255),density=False)

    # Find a mapping from the input pixels to the transformation function s
    # using the cumulative probability
    hstpdf1 = hst1 / numpy.float32(im1.size)
    cdf1 = hstpdf1.cumsum() # cumulative distribution function
    s = 255 * cdf1 # normalise

    # Find a mapping from the input pixels to the transformation function v
    # using the cumulative probability
    hstpdf2 = hst2 / numpy.float32(im2.size)
    cdf2 = hstpdf2.cumsum() # cumulative distribution function
    v = 255 * cdf2 # normalise

	# Compute the pixel mapping function
    k = 256
    F = numpy.zeros(shape=(k))

    for i in range(k):
        j = k - 1
        while True:
            F[i] = j
            j = j - 1
            if j < 0 or s[i] > v[j]:
                break

    # use linear interpolation of cdf to find new pixel values
    im3 = numpy.interp(im1.flatten(),bins1[:-1],F)

    return im3.reshape(im1.shape)

# Function to perform contrast enhancement using adaptive histogram equalization
#   Ref(s):
def histeqADAPT(im, radius=20):
    img = numpy.zeros(im.shape)

    for i in xrange(im.shape[0]):
        for j in xrange(im.shape[1]):
            block = im[max(i-radius,0):min(i+radius,im.shape[0]), max(j-radius,0):min(j+radius,im.shape[1])]
            hst,bins = numpy.histogram(block.flatten(),256, (0,255))
            # Calculate the cumulative histogram
            cdf = hst.cumsum()
            # Normalize the CDF
            cdf = 255 * cdf / cdf[-1]
            img[i][j] = cdf[im[i][j]]
    return img

# Function to perform contrast enhancement using bi-histogram equalization
#   Ref(s):
#   Kim, Y.-T., "Contrast enhancement using brightness preserving bi-histogram
#   equalization", IEEE Trans. on Consumer Electronics, Vol.43, pp.1-8 (1997)
#
def histeqBI(im,nbr_bins=256):
    img = numpy.zeros(im.shape)

    gmin = 0 #im.min()
    gmax = 255 #im.max()

    # Calculate the image mean
    Xm = math.ceil(im.mean())
    imFLAT = im.flatten()

    # Find values <= Xm, and calculate the histogram (Eq.7)
#    Xlow = imFLAT.compress((imFLAT<=imFLAT.mean()).flat)
    Xlow = imFLAT.compress((imFLAT<=Xm).flat)
    piv1 = Xm
    HSTlow,bins = numpy.histogram(Xlow,int(piv1+1),(0,piv1),density=False)

    # Calculate the probability density function of the sub-histogram (Eq.9)
    HSTlowPDF = HSTlow / numpy.float32(Xlow.size)
    # Derive the cumulative histogram (Eq.11)
    cL = HSTlowPDF.cumsum() # cumulative distribution function

    #xr = numpy.arange(0,Xm+1,1)
    #pylab.plot(xr,cL)
    #pylab.show()

    # Find values > Xm, and calculate the histogram (Eq.8)
    Xupp = imFLAT.compress((imFLAT>Xm).flat)
    piv2 = 255-Xm
    HSTupp,bins = numpy.histogram(Xupp,int(piv2),(piv1+1,255),density=False)

    # Calculate the probability density function of the sub-histogram (Eq.10)
    HSTuppPDF = HSTupp / numpy.float32(Xupp.size)
    # Derive the cumulative histogram (Eq.12)
    cU = HSTuppPDF.cumsum() # cumulative distribution function

    #xr = numpy.arange(Xm+1,256,1)
    #pylab.plot(xr,cU)
    #pylab.show()

	# Histogram equalization for each of the sub-histograms
    fL = gmin + (Xm - gmin) * cL      #(Eq.13)
    fU = Xm+1 + (gmax - (Xm+1)) * cU  #(Eq.14)

    # Convert to 0-255
    fL = fL.astype('uint8')
    fU = fU.astype('uint8')

    # Merge the two histograms to make transformation easier
    fALL = numpy.concatenate((fL,fU))

    #xr = numpy.arange(0,256,1)
    #pylab.plot(xr,fALL)
    #pylab.show()

	# Transform the original image using the new cumulative histogram
    for i in range(0,im.shape[0]):
        for j in range(0,im.shape[1]):
            img[i][j] = fALL[im[i][j]]

    # Return the modified image, and the cumulative histogram used to modify it
    return img, fALL

# Function to perform contrast enhancement using histogram hyperbolization
#   Ref(s):
#   Frei, W., "Image enhancement by histogram hyperbolization", Computer
#   Graphics and Image Processing, Vol.6, pp.286-294 (1977)
#
def histhyper(im,nbr_bins=256):
    img = numpy.zeros(im.shape)

    c_value = 0.5

    # Get the image histogram
    hst,bins = numpy.histogram(im.flatten(),nbr_bins,(0,255),density=False)

    # Normalize the histogram 0->1
    hstpdf = hst / numpy.float32(im.size)
    # Calculate the cumulative distribution function of the normalized histogram
    cdf = hstpdf.cumsum()

    hY = numpy.zeros(shape=(256))

    yLog = 1.0 + 1.0 / c_value

    # Ref-(Eq.4)
    for i in range(0,256):
        hY[i] = c_value * (math.exp(math.log(yLog) * cdf[i]) - 1)

    # Perform the histogram transformation
    for i in range(0,im.shape[0]):
        for j in range(0,im.shape[1]):
            img[i][j] = hY[im[i][j]]

#    xr = numpy.arange(0,256,1)
#    pylab.plot(xr,hY)
#    pylab.show()

    # Return the modified image, multiplied by 255 to normalize in the range 0->255
    return img*255

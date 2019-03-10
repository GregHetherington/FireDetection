# Library to perform image I/O and histogram plotting
#
# imread_gray    - Read in a grayscale image and return a numpy array
# imwrite_gray   - Write a grayscale numpy array into an image file (.PNG)
# imread_colour  - Read in a colour image and return three numpy arrays
# imwrite_colour - Write a colour numpy array into an image file (.PNG)
# plot_IMGhist   - Plot a histogram of an input image
# plot_hist      - Plot a histogram

import numpy as np
import PIL
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib
import pylab


# Function to read in a grayscale image and return as an 8-bit 2D array
def imread_gray(fname):
    img = PIL.Image.open(fname)
    return np.asarray(img)

# Function to write a grayscale image (.PNG) from an 8-bit 2D array
def imwrite_gray(fname,img):
    img_uint8 = np.uint8(img)
    imgSv = PIL.Image.fromarray(img_uint8,'L')
    imgSv.save(fname)

# Function to read in a grayscale image and return as an 8-bit 2D array
def imread_colour(fname):
    img = PIL.Image.open(fname)
    imgRGB = np.asarray(img)
    imCr = imgRGB[:,:,0]
    imCg = imgRGB[:,:,1]
    imCb = imgRGB[:,:,2]
    return imCr, imCg, imCb

# Function to write a grayscale image (.PNG) from an 8-bit 2D array
def imwrite_colour(fname,imgR,imgG,imgB):
    rgbArray = np.zeros((imgR.shape[0],imgR.shape[1],3), 'uint8')
    rgbArray[..., 0] = imgR
    rgbArray[..., 1] = imgG
    rgbArray[..., 2] = imgB
    imgSv = PIL.Image.fromarray(rgbArray)
    imgSv.save(fname)

# Function to display the histogram of an image
def plot_IMGhist(img,nbr_bins=256):
    # the histogram of the data
    plt.hist(img.flatten(),nbr_bins,(0,nbr_bins-1))

    plt.xlabel('Graylevels')
    plt.ylabel('No. Pixels')
    plt.title('Intensity Histogram')
    plt.grid(True)

    plt.show()

# Function to display an image histogram
def plot_hist(hst,nbr_bins=256):

    xr = np.arange(0,nbr_bins,1)
    pylab.plot(xr,hst)

    pylab.show()

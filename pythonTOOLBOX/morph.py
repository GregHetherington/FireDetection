# Library to process images using grayscale morphology
#
# morph_sharp        - Morphological image sharpening
# morph_toggleCE     - Morphological toggle contrast enhancement
# morph_CE           - Morphological image sharpening

import numpy
import math
import scipy.ndimage as nd
import pymorph

# Function to perform morphological image sharpening similar in 
# concept to morphological toggles for contrast enhancement
# USES: pymorph, numpy
# REF:   
#    Schavemaker, J.G.M. , Reinders, M.J.T., Gerbrands, J.J.,
#    Backer, E.E., "Image sharpening by morphological filtering",
#    Pattern Recognition, 2000, Vol.33, pp.997-1012.

def morph_sharp(im):
    
    img = numpy.zeros(im.shape,dtype=numpy.int32)
    
    # Choose a disk structuring element (3x3 disk)
    # Function could be modified to pass structuring element in
    se = pymorph.sedisk(r=1,dim=2)
    
    # Apply grayscale erosion
    Ie = pymorph.erode(im,se)
    # Apply grayscale dilation
    Id = pymorph.dilate(im,se)
    
    for i in range(0,im.shape[0]):
        for j in range(0,im.shape[1]):
            # Compute differences between original image and processed
            da = Id[i][j] - im[i][j]
            db = im[i][j] - Ie[i][j]
            
            if  da < db:
                img[i][j] = Id[i][j]
            elif da > db:
                img[i][j] = Ie[i][j]
            else:
                img[i][j] = im[i][j]
    return img
    
# Function to perform morphological toggle contrast enhancement
# USES: pymorph, numpy
# REF:   
#    Meyer & Serra

def morph_toggleCE(im):
    
    img = numpy.zeros(im.shape,dtype=numpy.int32)
    
    se = pymorph.sedisk(r=1,dim=2)
    
    Ie = pymorph.erode(im,se)
    Id = pymorph.dilate(im,se)

    for i in range(0,im.shape[0]):
        for j in range(0,im.shape[1]):
            da = Id[i][j] - im[i][j]
            db = im[i][j] - Ie[i][j]
            
            if  da < db:
                img[i][j] = Id[i][j]
            else:
                img[i][j] = Ie[i][j]

    return img
    
# Function to perform morphological contrast enhancement
# USES: pymorph, numpy
# REF:   
#    Soille, P., "A note on morphological contrast enhancement",
#    Ecole des Mines d'Ales & EERIE: Nimes Cedex, France, pp.1-7 (1997)

def morph_CE(im):
    
    img = numpy.zeros(im.shape,dtype=numpy.int32)
    
    se = pymorph.sedisk(r=1,dim=2)
    
    # Top-hat by opening
    THo = pymorph.openth(im,se)
    # Top-hat by closing
    THc = pymorph.closeth(im,se)

    for i in range(0,im.shape[0]):
        for j in range(0,im.shape[1]):
            newPixel = im[i][j] + THo[i][j] - THc[i][j]
            if newPixel > 255:
            	img[i][j] = 255
            else:
            	img[i][j] = newPixel
  
    return img
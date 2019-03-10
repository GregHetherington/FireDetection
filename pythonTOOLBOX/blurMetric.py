import numpy
import scipy
import scipy.ndimage.filters as nd

"""
Krotkov, E., Martin, J-P., "Range from focus", IEEE Conf. on 
Robotics and Automation. Proceedings. pp.1093-1098 (1986)
Note: the output (blur) implies -> 0 means blur, 0 -> means sharp
"""
def GRADfocus(img):

    Gx = nd.sobel(img, axis=0)
    Gy = nd.sobel(img, axis=1)

    Gmag = Gx**2 + Gy**2
    
    focusMeasure = numpy.mean(Gmag)
    
    return focusMeasure

"""
Pech-Pacheco, J.L., Cristobal, G., Martinez, J.C., Valdivia, J.F.,
"Diatom autofocusing in brightfield microscopy: a comparative study", 
15th International Conference on Pattern Recognition, pp.314-317 (2000).
Sobel-Tenengrad gradient magnitude variance
Note: the output (blur) implies -> 0 means sharp, 0 -> means blur
"""
def STGMVfocus(img):

    Gx = nd.sobel(img, axis=0) # horizontal
    Gy = nd.sobel(img, axis=1) # vertical

    # Calculate the gradient magnitude
    Gmag = numpy.hypot(Gx, Gy) 
    
    Gmean = numpy.mean(Gmag,dtype=numpy.float32)
    Gdiff = (Gmag - Gmean)**2
    focusMeasure = numpy.sum(Gdiff,dtype=numpy.float32)
    
    return focusMeasure

"""
Crete-Roffet F., Dolmiere T., Ladret P., Nicolas M., "The Blur Effect: 
Perception and estimation with a new no-reference perceptual blur metric", 
SPIE Electronic Imaging Symposium Conf Human Vision and Electronic Imaging (2007)
Note: the output (blur) is in [0,1]; 0 means sharp, 1 means blur
"""
def perblurMetric(img):
    
    imagFlt = img.astype(float)
    x,y = img.shape
    
    Hv = numpy.zeros((9,9),dtype=numpy.float)
    #Hv[:] = 1.0/9.0;
    Hv[4,:] = 1.0/9.0
    Hh = numpy.transpose(Hv)

    # blur the input image in vertical direction
    BLver = nd.convolve(img,Hv)
    # blur the input image in horizontal direction
    BLhor = nd.convolve(img,Hh)
    
    # variation of the input image (vertical direction)
    DFver = abs(img[:,0:x-1] - img[:,1:x])
    # variation of the input image (horizontal direction)
    DFhor = abs(img[0:y-1,:] - img[1:y,:])
    
    # variation of the blurred image (vertical direction)
    DBver = abs(BLver[:,0:x-1] - BLver[:,1:x])
    # variation of the blurred image (horizontal direction)
    DBhor = abs(BLhor[0:y-1,:] - BLhor[1:y,:])
    
    # difference between two vertical variations of 2 image (input and blurred)
    Tver = DFver - DBver
    # difference between two horizontal variations of 2 image (input and blurred)
    Thor = DFhor - DBhor
    
    Vver = numpy.maximum(0,Tver)
    Vhor = numpy.maximum(0,Thor)

    SDver = numpy.sum(DFver,dtype=numpy.float32)
    SDhor = numpy.sum(DFhor,dtype=numpy.float32)
        
    SVver = numpy.sum(Vver,dtype=numpy.float32)
    SVhor = numpy.sum(Vhor,dtype=numpy.float32)

    blurFver = (SDver-SVver)/SDver
    blurFhor = (SDhor-SVhor)/SDhor

    blur = numpy.maximum(blurFver,blurFhor)

    return blur

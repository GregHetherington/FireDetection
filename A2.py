import sys

curDir = sys.path[0]
sys.path.append(curDir + '/pythonTOOLBOX')

from imageIO import *
import imageIO

from PIL import Image
import cv2
import numpy as np
import time

def getCommandArgs(curDir):
    if(len(sys.argv) < 2):
        print('inncorrect number of command line args')
    inputImage = curDir + "/" + sys.argv[1]

    return inputImage

#Algorithm #1
def alg1(image):
    print("Running Algorithm1 ...\n")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    weaker = np.array([0,130,130])
    stronger = np.array([40,255,255])
    mask = cv2.inRange(hsv, weaker, stronger)
    res = cv2.bitwise_and(image,image, mask= mask)
    edges = cv2.Canny(mask,100,200)

    Z = image.reshape((-1,3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 8
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((image.shape))
    output = cv2.bitwise_and(res2,res2, mask= mask)

    print("... Finished Running Algorithm1\n")
    return output

#Algorithm #2
def alg2(image):
    print("Running Algorithm2 ...\n")

    blur = cv2.GaussianBlur(image, (21, 21), 0)
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2Lab )

    #Find the mean of the L* A* B* values
    L_list = []
    a_list = []
    b_list = []

    PixelArray = np.asarray(lab)
    for n, dim in enumerate(PixelArray):
        for num, row in enumerate(dim):
            L, a, b = row
            L_list.append(L)
            a_list.append(a)
            b_list.append(b)

    L_mean = np.mean(L_list)
    a_mean = np.mean(a_list)
    b_mean = np.mean(b_list)

    #Check all the Rules of the Algorithm
    mask = image.copy()

    #LAB
    for y in range(0, lab.shape[0]):
        for x in range(0, lab.shape[1]):
            L, a, b = lab[y, x]
            if(L >= L_mean and a >= a_mean and b >= b_mean and b >= a):
                mask[y, x] = 255
            else:
                mask[y, x] = 0

    lower = np.array([255, 255, 255], dtype="uint8")
    upper = np.array([255, 255, 255], dtype="uint8")
    maskR = cv2.inRange(mask, lower, upper)

    alg1 = cv2.bitwise_and(image,image ,mask = maskR)

    hsv = cv2.cvtColor(alg1, cv2.COLOR_BGR2HSV)

    #HSV
    lower = [6, 80, 150]
    upper = [35, 255, 255]
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    mask = cv2.inRange(hsv, lower, upper)

    output = cv2.bitwise_and(image,image ,mask = mask)

    print("... Finished Running Algorithm2\n")

    return output

# Main Thread
print('Welcome to Greg & Petar\'s Image Process Assignment #2: Spot the Fire\n')
inputImage = getCommandArgs(curDir)
print('Input Image: ' + inputImage)

#Get Image
srcBGR = cv2.imread(inputImage)

#Run Algorithm
start1 = time.time()
newImageA1 = alg1(srcBGR)
end1 = time.time()
newImageA2 = alg2(srcBGR)
end2 = time.time()

print("RunTime Analysis: ")
print("Alg 1: ")
print(end1-start1)
print("Alg 2: ")
print(end2-end1)

#Save Image
print("Saving Image: ...\n")
cv2.imwrite("new_fireout_algorithm_1.jpg", newImageA1)
cv2.imwrite("new_fireout_algorithm_2.jpg", newImageA2)

print("Done. exiting...")

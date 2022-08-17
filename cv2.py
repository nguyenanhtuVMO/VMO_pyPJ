import cv2
import numpy
import matplotlib.pyplot as plt

plt.imshow(im, cmap='gray')
plt.axis('off')
plt.show()	
image = cv2.imread ('xedinguocchieu_1.jpg')
blur_image = cv2.blur(image, (23, 23))

mask = numpy.zeros(image.shape, dtype = numpy.unit8)
mask = cv2.cvtColor(mask, cv2.COLOR_BG2GRAY)
cv2.square(mask, (150,100), 50, 255, -1)
front_image= cv2.bitwise_and(blur_image, blur_image, mask = mask)

mask_inv = cv2.bitwise_not(mask)
back_image = cv2.bitwise_and(image, image, mask = mask.inv)

cv2.imshow('Car', front_image + back_image)
cv2.waitKey(0)

def fd_haralick(image):
    # convert the image to gray
     gray = cv2.cvtColor(image, cv2.COLOR_BG2GRAY)
     haralick = mahotas.feature.haralick(gray).mean(axis = 0)
     # return the result
     return haralick

def fd_histogram(image, mask = None):
    # convert the image to HSV space
     image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
     hist = cv2.CalcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
     # normalize the histogram
     cv2.nomalize(hist, hist)
     return hist.flatten()

def fd_hu_moments(image):
    # convert the image to gray
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image).flatten)
    # return the result
    return feature
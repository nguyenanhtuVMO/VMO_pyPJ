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




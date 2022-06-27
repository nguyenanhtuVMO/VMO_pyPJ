from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

def fd_histogram(image, mask=None):
    # chuyển về không gian màu HSV
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize histogram
    cv2.normalize(hist, hist)
    return hist.flatten()

def fd_hu_moments(image):
    # chuyển về ảnh gray
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

def fd_haralick(image):
    # chuyển về ảnh gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick

# path to output
output_path = "D:\\project\\fruit-classification\\output\\"

# path to training data
train_path = "D:\\project\\fruit-classification\\dataset\\train\\"

# get the training labels
train_labels = os.listdir(train_path)
train_labels.sort()

# num of images per class
images_per_class = 400

# fixed-sizes for image
fixed_size = tuple((100, 100))

# bins for histogram
bins = 8

# empty lists to hold feature vectors and labels
global_features = []
labels = []

# feature-descriptor-1: Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.CL_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mt.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick
# feature-descriptor-3: Color Histogram
def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256] )
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()

# read image form each folder

# loop over the training data sub-folders
for training_name in train_labels:
    dir = os.path.join(train_path, training_name)
# get the current training label
current_label = training_name
# loop over the image in each sub-folder
for x in range(1, images_per_class+1):
    # get the image file name
    file = dir + "\\" + "Image ("+str(x) + ").jpg"
    print(file)

image = cv2.imread(file)
image = cv2.resize(image, fixed_size)

# Global Feature extraction
fv_hu_moments = fd_hu_moments(image)
fv_haralick   = fd_haralick(image)
fv_histogram  = fd_histogram(image)

# Concatenate global features

global_features = np.hstack([fv_histogram, fv_hu_moments, fv_haralick])

labels.append(current_label)
global_features.append(global_feature)

print("[STATUS] processed folder: {}".format(current_label))

print("[STATUS] completed Global Feature Extraction...")

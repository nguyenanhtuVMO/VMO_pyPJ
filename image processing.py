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

# get the overall feature vector size
print ("[STATUS] feature vector size {}".format(np.array(global_features).shape))

# get the overall training label size
print ("[STATUS] training Labels {}".format(np.array(labels).shape))
 
# encode the target labels
le = LabelEncoder()
target = le.git_transform(labels)

# normalize the feature vector in range (0-1)
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_feature = scaler.fit_transform(global_features)

# save the feature vector using HDF5
h5f_data = h5py.File(output_path+'data.h5', 'w')
h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))

h5f_label = h5py.File(output_path+'labels.h5', 'w')
h5f_label.create_dataset('dataset_1', data=np.array(target))

h5f_data.close()
h5f_label.close()

print("[STATUS] end of training..")
# path to oput
output_path = "D:\\project\\fruit-classification\\output\\"
# fixed-size for image
fixed_size = tuple((100, 100))
# no.of.tress for Random Forests
num_stress = 300
# bins for histogram
bin = 8
# num of image per class
images_per_class = 10
# import the feature vector and trained labels
h5f_data = h5py.File(output_path+ 'data.h5', 'r')
h5f_label = h5py.File(output_path+ 'data.h5', 'r')

global_features_string = h5f_data['dataset_1']
global_labels_string = h5f_data['dataset_1']

global_features = np.array(global_features_string)
global_labels = np.array(global_labels_string)

h5f_data.close()
h5f_label.close()

#feature-descriptor-1: Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

# feature-descriptor-2: Haralick Texture
def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick textture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick

# feature-descriptor-3: Color Histogram
def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # comute the color histogram
    hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()

# create the model - Random Forests
clf = randomForestClassifier(n_estimators=num_stress)
clf.fit(global_features, global_labels)

# path to test data
test_path = "D:\\project\\fruit-classification\\dataset\\test"
# get the training labels
test_labels = os.listdir(test_path)

# sort the training labels 
test_labels.sort()
print(tetest_labels)

# loop through the test images
test_features = []
test_result = []
for testing_name in test_labels :
    
    # join the training data path and each species training folder
    dir = os.path.join(test_path, testing_name)
    # get the curent training label
    current_label = testing_name
    # loop over the images in eache sub-folder
    for x in range(1, images_per_class+1):
        #get the image file name
        index = random.randint(1,150);
        file = dir + "\\" + "Image ("+str(index) + ").jpg"
        print(file)
        image = cv2.imread(file)
        image = cv2.resize(image, fixed_size)

        fv_hu_moments = fdfd_hu_moments(image)
        fv_haralick = fd_haralick(image)
        fv_histogram = fd_histogram(image)

        test_result.append(current_label)
        test_features.append(np.hstack([fv_histogram, fv_hu_moments, fv_haralick]))

# predict label of test image 
le = LabelEncoder()
y_result = le.fit_transform(test_result)
y_pred = clf.predict(test_features)
print(y_pred)
print("Result: ", (y_pred == y_result).tolist().count(True)/len(y_result))

def fd_hu_moments(image):
    # convert image to gray
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

def fd_haralick():
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick
global_features = []
lable = []

global_features = np.array(global_features_string)
global_labels = np.array(global_labels_string)

# creat the model - Random Forest.
clf = RandomForestClassifier(n_estimator=num_stress)
clf.fit(global_features, global_labels)
# path to test data
test_path = "D:\\project\\fruit-classification\\dataset\\test"
# get the training labels
test_labels = os.listdir(test_path)

#sort the training labels
test_labels.sort()
print(test_labels)

global_features = np.array(global_features_string)
global_labels = np.array(global_labels_string)
# loop through the test images'
test_features = []
test_labels = []
for testing_name in test_labels:
   # join the training data path and each species training folder
    dir = os.path.join(test_path, testing_name)
    # get the current training labels \
    current_label = testing_name
    # loop over the images in each sub-folder
    for x in range (1,image_per_class+1):
        # get the image file name 
        index = random.randint(1.150);
        file= dir+"\\" + "Image ("+str(index) + ").jpg"
        print(file)
        image = cv2.imread(file)
        image = cv2.resize(image, fixed_size)
        ########
        fv_hu_moments = fd_hu_moments(image)
        fv_histogram = fd_histogram(image)
        fv_haralick = fd_haralick(image)
        ########
        test_result.append(current_label)
        test_features.append(np.hstack([fv_histogram,fv_haralick,fv_moments]))

        # predict label of test image
        le = LabelsEncoder()
        y_result = le.fit_transform(test_result)
        y_pred = clf.predict(test_features)
        print(y_pred)
        print("Result: ", (y_pred == y_result).tolist().count(True)/len(y_result))

def fd_haralick(iamge) :
    # convert imgae to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.feature.haralick(gray).mean(axis = 0)
    # return the result
    return haralick

def fd_histogram(image, mask = none):
    # convert image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist = cv2.calcHist([image], [0,1,2], None , [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the result
    return hist.flatten()




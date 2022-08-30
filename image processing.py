from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion

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


global_features = []
labels = []


def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.CL_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

def fd_haralick(image):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   
    haralick = mt.features.haralick(gray).mean(axis=0)
    
    return haralick

def fd_histogram(image, mask=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256] )
    
    cv2.normalize(hist, hist)
   
    return hist.flatten()



# loop over the training data sub-folders
for training_name in train_labels:
    dir = os.path.join(train_path, training_name)

current_label = training_name
# loop over the image in each sub-folder
for x in range(1, images_per_class+1):
    
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


print ("[STATUS] feature vector size {}".format(np.array(global_features).shape))


print ("[STATUS] training Labels {}".format(np.array(labels).shape))
 

le = LabelEncoder()
target = le.git_transform(labels)
                                      
def detect_peaks(image):
   

    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2,2)

    #apply the local maximum filter; all pixel of maximal value 
    #in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood)==image
    #local_max is a mask that contains the peaks we are 
    #looking for, but also the background.
    #In order to isolate the peaks we must remove the background from the mask.

    #we create the mask of the background
    background = (image==0)

    #a little technicality: we must erode the background in order to 
    #successfully subtract it form local_max, otherwise a line will 
    #appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    #we obtain the final mask, containing only peaks, 
    #by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background

    return detected_peaks


#applying the detection and plotting results
for i, paw in enumerate(paws):
    detected_peaks = detect_peaks(paw)
    pp.subplot(4,2,(2*i+1))
    pp.imshow(paw)
    pp.subplot(4,2,(2*i+2) )
    pp.imshow(detected_peaks)

pp.show()

# save the feature vector using HDF5
h5f_data = h5py.File(output_path+'data.h5', 'w')
h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))

h5f_label = h5py.File(output_path+'labels.h5', 'w')
h5f_label.create_dataset('dataset_1', data=np.array(target))

h5f_data.close()
h5f_label.close()

print("[STATUS] end of training..")

output_path = "D:\\project\\fruit-classification\\output\\"

fixed_size = tuple((100, 100))

num_stress = 300

bin = 8

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
    
    
    dir = os.path.join(test_path, testing_name)
    
    current_label = testing_name
    
    for x in range(1, images_per_class+1):
      
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
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    
    return haralick
global_features = []
lable = []

global_features = np.array(global_features_string)
global_labels = np.array(global_labels_string)

# creat the model - Random Forest.
clf = RandomForestClassifier(n_estimator=num_stress)
clf.fit(global_features, global_labels)

test_path = "D:\\project\\fruit-classification\\dataset\\test"

test_labels = os.listdir(test_path)


test_labels.sort()
print(test_labels)

global_features = np.array(global_features_string)
global_labels = np.array(global_labels_string)

test_features = []
test_labels = []
found = ImageMultiply[waldo, ImageAdd[ColorConvert[pos, "GrayLevel"], .5]]
for testing_name in test_labels:
  
    dir = os.path.join(test_path, testing_name)
   
    current_label = testing_name

   
    for x in range (1,image_per_class+1):
        
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
   
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
    haralick = mahotas.feature.haralick(gray).mean(axis = 0)
    
    return haralick

def fd_histogram(image, mask = none):
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    hist = cv2.calcHist([image], [0,1,2], None , [bins, bins, bins], [0, 256, 0, 256, 0, 256])
   
    cv2.normalize(hist, hist)
   
    return hist.flatten()

def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature= cv2.HuMoments(cv2.MoMents(image)).flatten()
    return feature()




for x in range (1,  images_per_class):
     
    dir = os.path.join(train_path, training_name)
    
    currentcurrent_label = training_name
  
    for x in range(1, images_per_class+1):
       
        file =  dir + "\\" + "Image ("+str(x) + ").jpg"
        print(file)

       
        image = cv2.imread(file)
        image = cv2.resize(image, fixed_size)

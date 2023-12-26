import pandas as pd
import matplotlib.image as mpimg
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.feature import hog, local_binary_pattern
import warnings

from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=FutureWarning)


path = "C:/Users/Home/OneDrive/Bureau/SignLanguageDetection/signe_langauge/ML_data/asl_dataset/"

# Get a list of all the image files in the folder
img_files = os.listdir(path)
def preproces(path):
    # Load the image
    img = mpimg.imread(path)

    # Resize the image to 28x28
    img = cv2.resize(img, (28, 28))

    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Compute the HOG features
    hog_features, hog_image = hog(gray_img, visualize=True, block_norm='L2-Hys')
    return hog_features

def preprocess_SIFT(path):
    # Load the image
    image = cv2.imread(path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create SIFT object
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)

    return keypoints, descriptors
def preproces_LBP(path):
    # Load the image
    img = cv2.imread(path)

    # Resize the image to 28x28
    img = cv2.resize(img, (28, 28))

    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Define LBP parameters
    radius = 3
    n_points = 8 * radius
    METHOD = 'uniform'

    # Calculate LBP features
    lbp = local_binary_pattern(gray_img, n_points, radius, METHOD)

    # Normalize LBP histogram
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    hist = hist.astype('float32')

    return hist

def get_x_and_y():
    data = {}
    x = []
    y = []
    for im in img_files:
        data[im] = []
        imgs = os.listdir(path + im)
        for img_file in imgs:
            l = path + im + "/" + img_file
            #l=preproces_LBP(l)
            l = preproces(l)
            data[im].append(l)
            x.append(l)
            y.append(im)

    return {'x': x, 'y': y}

def get_data():

    # print(img_files)

    data = get_x_and_y()
    X = np.array(data['x'])

    Y = np.array(data['y'])
    # Combine X and Y into a single array
    data = np.column_stack((X, Y))

    # Shuffle the data
    #np.random.shuffle(data)

    # Split the shuffled data back into X and Y
    #X = data[:, :-1]

    #Y = data[:, -1]
    return [X,Y]

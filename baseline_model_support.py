import cv2
import numpy as np
from skimage import feature
from PIL import Image
import os
import tensorflow as tf
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi
from tensorflow.keras.applications.vgg19 import VGG19,preprocess_input
from keras.preprocessing import image
from tensorflow.keras.models import Model

models = {
    'none' :   None   ,
    'vgg16': (224, 224),
    'vgg19': (224, 224),
    'resnet50': (224, 224),
    'inceptionv3': (299, 299),
    'inceptionresnetv2': (299, 299),
    'xception': (299, 299),
    'efficientnetb0': (224, 224),
    'efficientnetb3': (300, 300)
}


def load_resized_images(model_name, folder_path):
    input_size = models[model_name]
    images = []
    for filename in sorted(os.listdir(folder_path), key=lambda x: int(os.path.splitext(x)[0])):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img = cv2.imread(os.path.join(folder_path, filename))
            if img is not None:
                # Convert BGR to RGB color space
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                resized_img = cv2.resize(img, input_size)
                images.append(resized_img)
    return np.array(images)



def color_histogram(image, bins=(4, 4, 2)):
    # Compute a 3D histogram in the RGB color space,
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    
    # Normalize the histogram so that images with the same content, 
    # but either scaled larger or smaller will have (roughly) the same histogram
    hist = cv2.normalize(hist, hist)

    # return our 3D histogram as a flattened array
    return hist.flatten()




def extract_lbp_features(image, P=8, R=1):
    # Convert the image to grayscale if it's a color image
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    lbp = feature.local_binary_pattern(image, P, R, method="uniform")
    
    # Build a histogram of LBP codes, but exclude the non-uniform bin.
    hist, _ = np.histogram(lbp, bins=np.arange(0, P + 3), range=(0, P + 2), density=True)

    # Normalize the histogram.
    hist = hist.astype("float")
    hist /= hist.sum()

    return hist




def compute_gist_descriptor(image):

    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Helper functions
    def compute_avg(img):
        r, c = img.shape
        chunks_row = np.split(np.array(range(r)), 4)
        chunks_col = np.split(np.array(range(c)), 4)
        grid_images = []
        for row in chunks_row:
            for col in chunks_col:
                grid_images.append(np.mean(img[np.min(row):np.max(row), np.min(col):np.max(col)]))
        return np.array(grid_images).reshape((4, 4))

    def power(image, kernel):
        image = (image - image.mean()) / image.std()
        return np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 +
                   ndi.convolve(image, np.imag(kernel), mode='wrap')**2)

    # Ensure the image is 0-1
    image = image / 255.0

    # Compute Gabor features
    results = []
    for theta in range(8):  # 8 orientations
        theta = theta / 8. * np.pi
        for frequency in (0.1, 0.2, 0.3, 0.4):  # 4 frequencies
            kernel = gabor_kernel(frequency, theta=theta)
            # Save power image for each filter
            results.append(power(image, kernel))

    # Compute the gist descriptor
    return np.array([compute_avg(power_image) for power_image in results]).reshape(512,)



def compute_subject_and_background(image):
    # Convert the image to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create a static saliency detector
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    
    # Compute the saliency map
    _, saliency_map = saliency.computeSaliency(gray)
    
    # Binarize the saliency map
    _, subject_mask = cv2.threshold(saliency_map, np.mean(saliency_map), 1, cv2.THRESH_BINARY)
    
    # Get the subject area
    subject_area = cv2.bitwise_and(gray, gray, mask=subject_mask.astype(np.uint8))
    
    # Invert the subject mask to get the background mask
    background_mask = cv2.bitwise_not(subject_mask)
    
    # Get the background area
    background = cv2.bitwise_and(gray, gray, mask=background_mask.astype(np.uint8))

    return subject_area, background


def compute_clarity_contrast_feature(image, subject_region, beta=0.2):

    # Compute FFT of the image and the subject region
    FI = np.fft.fft2(image)
    FR = np.fft.fft2(subject_region)

    # Compute the maximum absolute values
    max_FI = np.max(np.abs(FI))
    max_FR = np.max(np.abs(FR))

    # Compute MI and MR
    MI = np.where(np.abs(FI) > beta * max_FI)
    MR = np.where(np.abs(FR) > beta * max_FR)

    # Compute the clarity contrast feature
    fc = (len(MR[0]) / np.prod(subject_region.shape)) / (len(MI[0]) / np.prod(image.shape))

    return fc

def compute_hue_count_feature(image, beta=0.05):
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Compute a 20-bin histogram of the hue channel
    Hc = cv2.calcHist([hsv], [0], None, [20], [0, 180])
    
    # Compute m, the maximum histogram value
    m = np.max(Hc)
    
    # Compute Nc, the set of bins with values larger than βm
    Nc = np.where(Hc > beta * m)[0]
    
    # Compute the hue count feature
    f_l = 20 - len(Nc)
    
    return f_l

def compute_brightness_contrast_feature(subject_area, background):
    # Compute average lighting (brightness) of the subject area and the background
    Bf = np.mean(subject_area)
    Bb = np.mean(background)   
    # Compute lighting contrast feature
    f = np.log(Bf / Bb)
    return f


def compute_composition_feature(subject):
    # Calculate the centroid of the subject
    M = cv2.moments(subject)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    centroid = np.array([cX, cY])

    # Calculate the four intersection points based on the Rule of Thirds
    h, w = subject.shape[0], subject.shape[1]
    points = np.array([[w/3, h/3], [2*w/3, h/3], [w/3, 2*h/3], [2*w/3, 2*h/3]])

    # Compute the minimum distance between the centroid and the four points
    dists = np.sqrt(((centroid[0] - points[:, 0])**2 / w**2) + ((centroid[1] - points[:, 1])**2 / h**2))
    min_dist = np.min(dists)

    return min_dist



def compute_background_simplicity_feature(image):
    # Resize image pixels to be in range 0-15 instead of 0-255
    image_resized = (image / 16).astype(int)
    
    # Convert to 1D array
    flattened = image_resized.reshape(-1, 3)
    
    # Create histogram with 16*16*16 bins (4096 bins)
    hist = np.histogramdd(flattened, bins=(16, 16, 16))
    hist = hist[0].flatten()  # The histogramdd function returns a 2-tuple, and we need the first element
    
    # Compute the maximum count in the histogram
    hmax = np.max(hist)
    
    # Compute S, the indices where the condition is met
    S = np.where(hist >= 0.01 * hmax)[0]
    
    # Compute simplicity feature
    fs = (len(S) / 4096) * 100  # As a percentage
    return fs


# Load the VGG19 model
base_model = VGG19(weights='imagenet')

# Extract features from the 'fc2' layer(second to last layer with 4096 features)
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

def extract_features(img):
    # Load and preprocess the input image
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # Extract features
    features = model.predict(img)

    return features



# model and folder path
model_name = 'vgg16'
folder_path = 'model_size_pics/vgg19'

# Load resized images
images = load_resized_images(model_name, folder_path)


visual_features = []
color_entropy_earth = np.load("color_entropy_earth.npy")
color_entropy_pics = np.load("color_entropy_pics.npy")
# Loop through each image and compute the features




for image,item in zip(images,color_entropy_pics): 
    # Calculate subject area and background
    subject_area, background = compute_subject_and_background(image)
    
    # Compute the features
    color_hist = np.array(color_histogram(image)).flatten()
    lbp = np.array(extract_lbp_features(image)).flatten()
    gist = np.array(compute_gist_descriptor(image)).flatten()
    clarity_contrast = np.array(compute_clarity_contrast_feature(image, subject_area)).flatten()
    hue_count = np.array([compute_hue_count_feature(image)])
    brightness_contrast = np.array([compute_brightness_contrast_feature(subject_area, background)])
    composition = np.array([compute_composition_feature(subject_area)])
    background_simplicity = np.array([compute_background_simplicity_feature(image)])
    deep_learning = np.array(extract_features(image)).flatten()
    item = item.flatten()

# Create a single 1D feature vector
    features = np.concatenate([color_hist, lbp, gist, clarity_contrast, hue_count, item, brightness_contrast, composition, background_simplicity, deep_learning])

     
  
    
    visual_features.append(features)
    
    # Now you have the features and can use them for your next steps (e.g., feed them into a machine learning model)
    # For demonstration, we're just printing them
    print('Color Histogram:', color_hist.shape)
    print('LBP Features:', lbp.shape)
    print('Gist Descriptor', gist.shape)
    print('Clarity Contrast Feature:', clarity_contrast)
    print('Hue Count Feature:', hue_count)
    print('Color_entropy:', item.shape)
    print('Brightness Contrast Feature:', brightness_contrast)
    print('Composition Feature:', composition)
    print('Background Simplicity Feature:', background_simplicity)
    print('vgg19 features:', deep_learning.shape)
    print('------------------------------------')
   

    
print(np.array(visual_features).shape)

np.save("visual_features_pics.npy",visual_features)
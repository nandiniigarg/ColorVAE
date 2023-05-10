import os
import numpy as np
from PIL import Image
from tensorflow.keras.utils import img_to_array
import matplotlib.pyplot as plt
import cv2
SIZE=160
def convert_images_to_bw(folder_path):
    image_array = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # Filter image files
            image_path = os.path.join(folder_path, filename)

            # Open image and convert to black and white
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (SIZE, SIZE))
            img = img.astype('float32') / 255.0
            # Convert image to numpy array
            image_array.append(img_to_array(img))

    return image_array

def read_color(folder_path):
    image_array = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # Filter image files
            image_path = os.path.join(folder_path, filename)
            # Open image and convert to black and white
            img = cv2.imread(image_path)
            # Convert image to numpy array
            img = cv2.resize(img, (SIZE, SIZE))
            img = img.astype('float32') / 255.0
            image_array.append(img_to_array(img))

    return image_array


# Convert images to black and white and save as array
train_gray_image = convert_images_to_bw('afhq/combine')
train_color_image = read_color('afhq/combine')
print("Trainging data loaded")
from sklearn.utils import resample
train_gray_image, train_color_image = resample(train_gray_image, train_color_image)

test_color_image = read_color('afhq/combine_val')
test_gray_image = convert_images_to_bw('afhq/combine_val')

test_color_image, test_gray_image = resample(test_color_image, test_gray_image)

print("Testing data loaded")

for i in range(45, 48):
    color = train_color_image[i]
    grayscale = train_gray_image[i]
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 3, 1)
    plt.title('Color Image', color='green', fontsize=20)
    plt.imshow(color)
    plt.subplot(1, 3, 2)
    plt.title('Grayscale Image ', color='black', fontsize=20)
    plt.imshow(grayscale)

    plt.show()

import cv2
import numpy as np
from skimage import exposure
import os

DATASET_PATH = "cropped"
ALPHA = 1.05
RADIUS = 5
TARGET_IMAGE_PATH = "target_image.jpg"

img_list = os.listdir(DATASET_PATH)
img_list = [os.path.join(DATASET_PATH, path) for path in img_list]

# Calculate the RGB grid for the target image and display grid points
target_img = cv2.imread(TARGET_IMAGE_PATH)


# Display the target image with grid points
# Process the rest of the images
for img_path in img_list:
    
    img = cv2.imread(img_path)
    # print(img.shape[:2][::-1])
    
    target_img = cv2.resize(target_img, img.shape[:2][::-1])
    matched = exposure.match_histograms(img, target_img, channel_axis=2)
    cv2.imshow("Original", img)
    cv2.imshow("Target", target_img)
    cv2.imshow("Adjusted Image", matched)

    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break


cv2.destroyAllWindows()

import cv2
import numpy as np
import os

DATASET_PATH = "image"

img_list = os.listdir(DATASET_PATH)
img_list = [os.path.join(DATASET_PATH, path) for path in img_list]
# print(*img_list, sep="\n")

for img_path in img_list:
    
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    img = cv2.resize(img, (int(w/2), int(h/2)), interpolation=cv2.INTER_AREA)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur_img = cv2.GaussianBlur(img, (7,7), 0)
    edges_mask = cv2.Canny(blur_img, threshold1=100, threshold2=100)

    cv2.imshow("Original", img)
    cv2.imshow("Blurred", blur_img)
    cv2.imshow("Edges", edges_mask)

    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()

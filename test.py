import cv2
import numpy as np
import os

DATASET_PATH = "image"

img_list = os.listdir(DATASET_PATH)
img_list = [os.path.join(DATASET_PATH, path) for path in img_list]

for img_path in img_list:
    
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    img = cv2.resize(img, (int(w/2), int(h/2)), interpolation=cv2.INTER_AREA)

    blur_img = cv2.GaussianBlur(img, (7, 7), 0)
    edges_mask = cv2.Canny(blur_img, threshold1=100, threshold2=100)
    contours, _ = cv2.findContours(edges_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    blank_image = np.zeros(img.shape, dtype=np.uint8)

    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # If the polygon has 4 vertices, we assume it's a rectangle
        if len(approx) == 4:
            # Filter by area (ignore small areas)
            area = cv2.contourArea(approx)
            if area < 100:  # You can adjust this threshold value based on your needs
                continue

            # # Filter by aspect ratio (ensure the rectangle isn't too skewed)
            # x, y, w, h = cv2.boundingRect(approx)
            # aspect_ratio = w / float(h)
            # if aspect_ratio < 0.8 or aspect_ratio > 1.2:  # Adjust range for stricter checks
            #     continue

            # Draw the clean and big rectangle on the blank image
            cv2.drawContours(blank_image, [approx], 0, (0, 255, 0), 2)

    cv2.imshow("Original", img)
    cv2.imshow("Contours", blank_image)
    cv2.imshow("Edges", edges_mask)

    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()

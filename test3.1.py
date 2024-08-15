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
    
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')
    found_rectangle = False
    rotation_angle = 0

    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # If the polygon has 4 vertices, we assume it's a rectangle
        if len(approx) == 4:
            # Filter by area (ignore small areas)
            area = cv2.contourArea(approx)
            if area < 100:
                continue
            
            # Draw the clean and big rectangle on the blank image
            cv2.drawContours(blank_image, [approx], 0, (0, 255, 0), 2)

            # Find the bounding box of the rectangle
            x, y, w, h = cv2.boundingRect(approx)
            rect_min_x, rect_max_x = x, x + w
            rect_min_y, rect_max_y = y, y + h

            # Update min and max coordinates
            min_x = min(min_x, rect_min_x)
            max_x = max(max_x, rect_max_x)
            min_y = min(min_y, rect_min_y)
            max_y = max(max_y, rect_max_y)
            
            if not found_rectangle:
                # Find the orientation of the first rectangle
                rect = cv2.minAreaRect(contour)
                rotation_angle = rect[2] - 90  # Adjust angle if needed
                found_rectangle = True
    
    if found_rectangle:
        # Calculate the enclosing rectangle's dimensions
        enclosing_width = max_x - min_x
        enclosing_height = max_y - min_y
        
        # Define the enclosing rectangle corners
        top_left = (min_x, min_y)
        bottom_right = (max_x, max_y)
        
        # Rotate the enclosing rectangle
        center = ((min_x + max_x) // 2, (min_y + max_y) // 2)
        M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        
        rotated_image = cv2.warpAffine(img, M, (w, h))
        rotated_blank_image = np.zeros(rotated_image.shape, dtype=np.uint8)
        
        # Rotate the enclosing rectangle corners
        rect_corners = np.array([
            [min_x, min_y],
            [min_x, max_y],
            [max_x, max_y],
            [max_x, min_y]
        ], dtype=np.float32)

        rotated_corners = cv2.transform(np.array([rect_corners]), M)[0]
        
        # Draw the rotated enclosing rectangle
        rotated_contours_image = np.zeros(rotated_image.shape, dtype=np.uint8)
        cv2.polylines(rotated_contours_image, [rotated_corners.astype(np.int32)], isClosed=True, color=(255, 0, 0), thickness=2)
        
        # Display results
        cv2.imshow("Original", img)
        cv2.imshow("Contours", blank_image)
        cv2.imshow("Edges", edges_mask)
        cv2.imshow("Rotated Enclosing Rectangle", rotated_contours_image)
    
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()

import cv2
import numpy as np
import os

DATASET_PATH = "image"
MARGIN = 0.04


img_list = os.listdir(DATASET_PATH)
img_list = [os.path.join(DATASET_PATH, path) for path in img_list]

for img_path in img_list:
    
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    blur_img = cv2.GaussianBlur(img, (7, 7), 0)
    edges_mask = cv2.Canny(blur_img, threshold1=100, threshold2=100)
    contours, _ = cv2.findContours(edges_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    blank_image = np.zeros(img.shape, dtype=np.uint8)
    
    found_rectangle = False
    angle = 0

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

            # Find the orientation of the first rectangle
            if not found_rectangle:
                found_rectangle = True
                rect = cv2.minAreaRect(contour)
                angle = rect[2] - 90

    if found_rectangle:
        # Rotate the original image
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(img, M, (w, h))
        
        # Rotate the contour image
        rotated_blank_image = cv2.warpAffine(blank_image, M, (w, h))

        # Find the new contours on the rotated contour image
        rotated_contours, _ = cv2.findContours(cv2.cvtColor(rotated_blank_image, cv2.COLOR_BGR2GRAY), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize extreme coordinates for enclosing rectangle
        min_x, max_x = w, 0
        min_y, max_y = h, 0

        # Update extreme coordinates based on rotated contours
        for contour in rotated_contours:
            for point in contour:
                x, y = point[0]
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
    
        # Compute the enclosing rectangle
        width = max_x - min_x
        height = max_y - min_y

        max_x += int(width*MARGIN)
        max_y += int(height*MARGIN)
        min_x -= int(width*MARGIN)
        min_y -= int(height*MARGIN)
        bottom_left = (int(min_x), int(min_y))
        top_right = (int(max_x), int(max_y))
        
        # Draw the enclosing rectangle on the rotated contour image
        cv2.rectangle(rotated_blank_image, bottom_left, top_right, (255, 0, 0), 2)
        
        # Display the rotated images
        cv2.imshow("Rotated Image", rotated_image)
        cv2.imshow("Rotated Contours", rotated_blank_image)
    else:
        # If no rectangle was found, just display the original images
        cv2.imshow("Original", img)
        cv2.imshow("Contours", blank_image)
        # Crop the region of interest (ROI) from the rotated image

    roi = rotated_image[min_y:max_y, min_x:max_x]

    # Save the cropped image
    cropped_image_path = os.path.join("cropped", f"cropped_{os.path.basename(img_path)}")
    cv2.imwrite(cropped_image_path, roi)

    # Optionally, display the cropped image
    cv2.imshow("Cropped Image", roi)

    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()

import cv2
import numpy as np
import os

DATASET_PATH = "cropped"
ALPHA = 1.05
RADIUS = 5

img_list = os.listdir(DATASET_PATH)
img_list = [os.path.join(DATASET_PATH, path) for path in img_list]

def get_average_rgb(img, center, radius):
    y, x = center
    # Define the square region around the centroid within the given radius
    y_min = max(0, y - radius)
    y_max = min(img.shape[0], y + radius + 1)
    x_min = max(0, x - radius)
    x_max = min(img.shape[1], x + radius + 1)
    
    # Extract the region and calculate the average RGB values
    region = img[y_min:y_max, x_min:x_max]
    avg_rgb = np.mean(region, axis=(0, 1))  # Average over height and width
    
    return avg_rgb

for img_path in img_list:
    
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    blur_img = cv2.GaussianBlur(img, (7, 7), 0)
    edges_mask = cv2.Canny(blur_img, threshold1=100, threshold2=100)
    contours, _ = cv2.findContours(edges_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    blank_image = np.zeros(img.shape, dtype=np.uint8)
    
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area < 100:
                continue

            cv2.drawContours(blank_image, [approx], 0, (0, 255, 0), 2)
    
    # Grid settings
    rows, cols = 4, 6
    row_spacing = int((h // (rows + 1)) * ALPHA)
    col_spacing = int((w // (cols + 1)) * ALPHA)
    
    # Initialize matrix to store average RGB values
    rgb_matrix = []

    # Draw grid points on the contour image and calculate average RGB
    for row in range(1, rows + 1):
        row_rgb_values = []
        if row == 2:
            continue
        for col in range(1, cols + 1):
            center_x = col * col_spacing
            center_y = row * row_spacing
            avg_rgb = get_average_rgb(img, (center_y, center_x), RADIUS)
            row_rgb_values.append(avg_rgb)  # Save the average RGB values
            cv2.circle(blank_image, (center_x, center_y), 5, (255, 0, 0), -1)
        
        rgb_matrix.append(row_rgb_values)
    
    # Convert rgb_matrix to a NumPy array for easier handling
    rgb_matrix = np.array(rgb_matrix)
    
    print("RGB Matrix:")
    print(rgb_matrix)
    
    cv2.imshow("Original", img)
    cv2.imshow("Contours", blank_image)
    cv2.imshow("Edges", edges_mask)

    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()

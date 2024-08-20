import cv2
import numpy as np
import os

DATASET_PATH = "cropped"
ALPHA = 1.05
RADIUS = 5
TARGET_IMAGE_PATH = "target_image.jpg"

img_list = os.listdir(DATASET_PATH)
img_list = [os.path.join(DATASET_PATH, path) for path in img_list]

def get_average_rgb(img, center, radius):
    y, x = center
    y_min = max(0, y - radius)
    y_max = min(img.shape[0], y + radius + 1)
    x_min = max(0, x - radius)
    x_max = min(img.shape[1], x + radius + 1)
    
    region = img[y_min:y_max, x_min:x_max]
    # avg_rgb = np.mean(region, axis=(0, 1)).astype(int)
    avg_rgb = np.mean(region, axis=(0, 1))
    
    return avg_rgb

# Calculate the RGB grid for the target image and display grid points
target_img = cv2.imread(TARGET_IMAGE_PATH)
target_h, target_w = target_img.shape[:2]

# Grid settings
rows, cols = 4, 6
row_spacing = int((target_h // (rows + 1)) * ALPHA)
col_spacing = int((target_w // (cols + 1)) * ALPHA)

target_rgb_matrix = []

for row in range(1, rows + 1):
    row_rgb_values = []
    for col in range(1, cols + 1):
        center_x = col * col_spacing
        center_y = row * row_spacing
        avg_rgb = get_average_rgb(target_img, (center_y, center_x), RADIUS)
        row_rgb_values.append(avg_rgb)
        cv2.circle(target_img, (center_x, center_y), 5, (0, 0, 255), -1)  # Draw grid point on target image
    target_rgb_matrix.append(row_rgb_values)

# Convert to NumPy array for easier handling
target_rgb_matrix = np.array(target_rgb_matrix)

print("Target image: \n", target_rgb_matrix)

# Display the target image with grid points
cv2.imshow("Target Image with Grid Points", target_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Process the rest of the images
for img_path in img_list:
    
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    blur_img = cv2.GaussianBlur(img, (7, 7), 0)
    edges_mask = cv2.Canny(blur_img, threshold1=100, threshold2=100)
    contours, _ = cv2.findContours(edges_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    blank_image = np.zeros(img.shape, dtype=np.uint8)
    
    rgb_matrix = []

    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area < 100:
                continue

            cv2.drawContours(blank_image, [approx], 0, (0, 255, 0), 2)

    row_spacing = int((h // (rows + 1)) * ALPHA)
    col_spacing = int((w // (cols + 1)) * ALPHA)
    
    for row in range(1, rows + 1):
        row_rgb_values = []
        for col in range(1, cols + 1):
            center_x = col * col_spacing
            center_y = row * row_spacing
            avg_rgb = get_average_rgb(img, (center_y, center_x), RADIUS)
            row_rgb_values.append(avg_rgb)
            cv2.circle(blank_image, (center_x, center_y), 5, (255, 0, 0), -1)
        
        rgb_matrix.append(row_rgb_values)

    print("Image Matrix (R, G, B):")
    rgb_matrix = np.array(rgb_matrix)
    print(rgb_matrix)
    
    # # Calculate the variance between the target image RGB grid and the detected RGB matrix
    # variance_matrix = np.var(target_rgb_matrix - rgb_matrix, axis=(0, 1))
    variance_matrix = cv2.absdiff(target_rgb_matrix, rgb_matrix)
    
    print("Variance Matrix (R, G, B):")
    print(variance_matrix)

    avg_diff = np.mean(variance_matrix, axis=(0, 1))
    print("Average Matrix (R, G, B):")
    print(avg_diff)
    
    adjusted_img = img.copy().astype(np.float32)
    # Adjust each channel
    adjusted_img[:, :, 0] += avg_diff[0]  # Blue channel
    adjusted_img[:, :, 1] += avg_diff[1]  # Green channel
    adjusted_img[:, :, 2] += avg_diff[2]  # Red channel
    adjusted_img = np.clip(adjusted_img, 0, 255).astype(np.uint8)

    cv2.imshow("Original", img)
    cv2.imshow("Contours", blank_image)
    cv2.imshow("Edges", edges_mask)
    cv2.imshow("Adjusted Image", adjusted_img)

    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()

import cv2
import numpy as np
# Load the image
image = cv2.imread('img.png')

# Apply hand detection or segmentation algorithm to get the hand region coordinates
# Define the hand region coordinates (x, y, width, height)
hand_x = 100
hand_y = 100
hand_width = 200
hand_height = 250



def region_growing(image, seed_point, tolerance):
    # Create an empty mask to store the segmented region
    mask = np.zeros_like(image, dtype=np.uint8)

    # Get the seed point coordinates
    seed_x, seed_y = seed_point

    # Get the seed point color
    seed_color = image[seed_y, seed_x]

    # Define the connectivity (8-connectivity for neighbors)
    connectivity = 8

    # Define the queue for pixel traversal
    queue = []
    queue.append((seed_x, seed_y))

    # Process until the queue is empty
    while len(queue) > 0:
        # Pop the first pixel from the queue
        x, y = queue.pop(0)

        # Check if the pixel is already visited
        if mask[y, x] == 255:
            continue

        # Get the current pixel color
        current_color = image[y, x]

        # Check the color similarity with the seed color
        color_diff = np.abs(current_color - seed_color)
        if np.all(color_diff <= tolerance):
            # Add the pixel to the segmented region
            mask[y, x] = 255

            # Add the neighbors to the queue
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if dx == 0 and dy == 0:
                        continue
                    nx = x + dx
                    ny = y + dy
                    # Check the boundary conditions
                    if nx >= 0 and ny >= 0 and nx < image.shape[1] and ny < image.shape[0]:
                        queue.append((nx, ny))

    return mask


# Load the image
image = cv2.imread('c_person.jpg')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply region growing segmentation on the grayscale image
seed_point = (100, 100)  # Set the seed point coordinates
tolerance = 20  # Set the tolerance for color similarity
segmented_region = region_growing(gray_image, seed_point, tolerance)

# Display the segmented region
cv2.imshow('Segmented Region', segmented_region)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Crop the image to include only the hand region
#hand_image = image[hand_y:hand_y+hand_height, hand_x:hand_x+hand_width]

# Display the cropped hand image
#cv2.imshow('Hand Image', hand_image)

#cv2.waitKey(0)
#cv2.destroyAllWindows()

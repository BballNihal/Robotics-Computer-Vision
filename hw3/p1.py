import os
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

# Define a function to resize an image
def resize_image(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dimensions = (width, height)
    resized_image = cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA)
    return resized_image

# Define the checkerboard size
checkerboard_size = (15, 10)  # (width, height)
square_size = 1.0  # Size of a square in your defined unit (e.g., cm or inches)

# Prepare object points (0,0,0), (1,0,0), ..., (6,5,0)
objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2) * square_size

# Prepare to store object points and image points
objpoints = []  # 3D point in real-world space
imgpoints = []  # 2D points in image plane

# Load images
images = glob.glob(r'C:\Users\nihal\OneDrive\Desktop\.vscode\CompVision\hw3\images\im*.jpg')  # Change to your path and file format

# Create 'processed' directory if it doesn't exist
output_dir = r'C:\Users\nihal\OneDrive\Desktop\.vscode\CompVision\hw3\processed'
os.makedirs(output_dir, exist_ok=True)

for img_path in images:
    img = cv2.imread(img_path)
    img = resize_image(img, scale_percent=20)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 1: Increase contrast
    gray_eq = cv2.equalizeHist(gray)

    # Display the processed images for debugging
    cv2.imshow('Processed Image', gray_eq)
    cv2.waitKey(500)

    # Step 3: Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray_eq, checkerboard_size, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
        corner_image = img.copy()
        cv2.drawChessboardCorners(corner_image, checkerboard_size, corners, ret)
        
        # Change output path to save in the processed folder
        output_path = os.path.join(output_dir, os.path.basename(img_path).replace('.jpg', '_processed_corners.jpg'))
        cv2.imwrite(output_path, corner_image)
        print(f'Saved: {output_path}')
    else:
        print(f'Checkerboard not found in: {img_path}')

cv2.destroyAllWindows()

# Calibrate camera
ret, K, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Print results
print("Camera Matrix (K):")
print(K)
print("\nRotation Vectors (R):")
for rvec in rvecs:
    print(rvec)
print("\nTranslation Vectors (t):")
for tvec in tvecs:
    print(tvec)

# Convert rotation vectors to rotation matrices
R_matrices = []
camera_positions = []

for rvec, tvec in zip(rvecs, tvecs):
    R, _ = cv2.Rodrigues(rvec)
    R_matrices.append(R)
    camera_positions.append(-R.T @ tvec)  # Camera position in world coordinates

# Extract x, y, z coordinates, ensure each position is in the expected format
X = [pos[0] for pos in camera_positions]
Y = [pos[1] for pos in camera_positions]
Z = [pos[2] for pos in camera_positions]

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X, Y, Z, c='r', marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Camera Positions in World Coordinates')
plt.show()

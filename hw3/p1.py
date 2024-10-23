#PROBLEM 1

# This code first loads and pinpoints the corners on the board from 20 different images. If the board can't be detected, the image is skipped.To make detection easier
# I scale the image size down, increase contrast, and greyscale it. The changes to K matrix from rescaling it is then reversed after calibration is done.
# Then I use open cv to calibrate the camera based on the detected corners from different angles and the according 2D and 3D points. This gives the K matrix.
# The rotation and translation vectors are then extracted and printed along with the K matrix. Finally, the camera positions for all 20 images are located and plotted.

import os
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#resize function
def resize_image(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dimensions = (width, height)
    resized_image = cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA)
    return resized_image

checkerboard_size = (15, 10)  
square_size = 1.0  

#object points for the checkerboard
objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2) * square_size

objpoints = []  
imgpoints = []  

images = glob.glob(r'C:\Users\nihal\OneDrive\Desktop\.vscode\CompVision\hw3\images\im*.jpg')  
output_dir = r'C:\Users\nihal\OneDrive\Desktop\.vscode\CompVision\hw3\processed'
os.makedirs(output_dir, exist_ok=True)

for img_path in images:

    #rescale, greyscale, contrast
    img = cv2.imread(img_path)
    # img = resize_image(img,50)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)

    cv2.imshow('Processed Image', gray_eq)
    cv2.waitKey(500)

    ret, corners = cv2.findChessboardCorners(gray_eq, checkerboard_size, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
        corner_image = img.copy()
        cv2.drawChessboardCorners(corner_image, checkerboard_size, corners, ret)
        
        output_path = os.path.join(output_dir, os.path.basename(img_path).replace('.jpg', '_processed_corners.jpg'))
        cv2.imwrite(output_path, corner_image)
        print(f'Saved: {output_path}')
    else:
        print(f'Checkerboard not found in: {img_path}')

cv2.destroyAllWindows()

#calibration
ret, K, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

#offset rescaling
# K = K*2
# Print K and rotation / translation matrices
print("Camera Matrix (K):")
print(K)
print("\nRotation Vectors (R):")
for rvec in rvecs:
    print(rvec)
print("\nTranslation Vectors (t):")
for tvec in tvecs:
    print(tvec)

R_matrices = []
camera_positions = []
viewing_vectors = []

#camera pos and viewing vectors
for rvec, tvec in zip(rvecs, tvecs):
    R, _ = cv2.Rodrigues(rvec)
    R_matrices.append(R)
    
    camera_position = -R.T @ tvec
    camera_positions.append(camera_position)
    
    viewing_direction = R.T @ np.array([0, 0, -1])
    viewing_vectors.append(viewing_direction)

#camera positions
X = [pos[0] for pos in camera_positions]
Y = [pos[1] for pos in camera_positions]
Z = [-pos[2] for pos in camera_positions]  

#veiwing vectors
U = [vec[0] for vec in viewing_vectors]
V = [vec[1] for vec in viewing_vectors]
W = [-vec[2] for vec in viewing_vectors]  

#ALL PLOTTING BELOW
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#camera postions
ax.scatter(X, Y, Z, c='r', marker='o', label='Camera Positions')
scale_factor = -3  

#camera viewing vectors
for i in range(len(camera_positions)):
    wtc = camera_positions[i]
    camera_direction = viewing_vectors[i]
    ax.quiver(
        wtc[0], wtc[1], -wtc[2], 
        camera_direction[0] * scale_factor, 
        camera_direction[1] * scale_factor, 
        -camera_direction[2] * scale_factor, 
        color='b'
    )

#Plot the checkerboard
corner_points = objp[[0, checkerboard_size[0] - 1, -1, -(checkerboard_size[0])]]  # 4 corners

ax.plot([corner_points[0, 0], corner_points[1, 0], corner_points[2, 0], corner_points[3, 0], corner_points[0, 0]],
        [corner_points[0, 1], corner_points[1, 1], corner_points[2, 1], corner_points[3, 1], corner_points[0, 1]],
        [0, 0, 0, 0, 0],  
        color='g', label='Checkerboard')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Camera Positions, Viewing Vectors, and Checkerboard')

plt.legend()
plt.show()

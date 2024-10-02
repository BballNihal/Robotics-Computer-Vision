#PROBLEM 1

#First I use the matplotlib library to grayscale and display the input image
#Then plt.input is used to get the corners of the homography input
#For the output bounds, I have chosen it to be the same size as the input image
#Then these points are sent the the homography function which actually performs the homography via the equation from class.
#The output image is finally created and displayed side by side with the input

import numpy as np
import cv2
from skimage import transform
import matplotlib.pyplot as plt

def estimate_homography(points1, points2):
    A = []
    for i in range(len(points1)):
        x, y = points1[i][0], points1[i][1]
        u, v = points2[i][0], points2[i][1]
        A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
        A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
    A = np.array(A)
    U, S, Vh = np.linalg.svd(A)
    L = Vh[-1, :] / Vh[-1, -1]
    H = L.reshape(3, 3)
    return H

def warp_image(image, H, output_shape):
    return transform.warp(image, np.linalg.inv(H), output_shape=output_shape)

def main():

    side_view = cv2.imread('building.jpg')
    side_view_gray = cv2.cvtColor(side_view, cv2.COLOR_BGR2GRAY)
    
    plt.imshow(side_view_gray, cmap='gray')
    plt.title('Select points clockwise starting at top left')
    points1 = plt.ginput(4)
    plt.close()

    h, w = side_view_gray.shape
    points2 = np.array([[0, 0], [w, 0], [w, h], [0, h]])

    H = estimate_homography(points1, points2)
    print("Estimated Homography:\n", H)

    warped_image = warp_image(side_view_gray, H, (h, w))

    plt.subplot(1, 2, 1)
    plt.imshow(side_view_gray, cmap='gray')
    plt.title('Side View')
    plt.subplot(1, 2, 2)
    plt.imshow(warped_image, cmap='gray')
    plt.title('Frontal View')
    plt.show()

if __name__ == "__main__":
    main()

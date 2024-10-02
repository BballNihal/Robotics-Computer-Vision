import numpy as np
from numpy.linalg import svd
import matplotlib.pyplot as plt

# Create X and Y coordinates for ellipse
a, b = 5, 100
t = np.arange(0, 100, 0.01)

x = a * np.sin(t)
y = b * np.cos(t)

# Display the original ellipse
plt.plot(x, y, '*-', label="Original Ellipse")
plt.title("Original Ellipse")
plt.show()

# Rotate the points of the ellipse
rotation_angle = np.pi / 3  # 60 degrees
cc = np.cos(rotation_angle)
ss = np.sin(rotation_angle)

R = np.array([[cc, ss],
              [-ss, cc]])

pts = np.stack((x, y))
rpts = np.dot(R, pts)

# Display the rotated ellipse
plt.plot(rpts[0, :], rpts[1, :], '*-', label="Rotated Ellipse")
plt.title("Rotated Ellipse")
plt.show()

# Use SVD to compute the orientation of the rotated ellipse
U, S, Vt = svd(rpts.T)
major_axis = U[:, 0]
est_rot_angle = np.arctan2(major_axis[1], major_axis[0])
est_rot_angle_deg = np.degrees(est_rot_angle)

print(f"Estimated rotation angle (radians): {est_rot_angle}")
print(f"Estimated rotation angle (degrees): {est_rot_angle_deg}")

# Now rotate original points back based on the estimated rotation angle
est_cc = np.cos(est_rot_angle)
est_ss = np.sin(est_rot_angle)

R_est = np.array([[est_cc, est_ss],
                  [-est_ss, est_cc]])

est_rpts = np.dot(R_est, pts)

# Plot both the original rotated points and the estimated rotated points
plt.plot(rpts[0, :], rpts[1, :], 'b*', alpha=0.5, label="Original Rotated Ellipse")
plt.plot(est_rpts[0, :], est_rpts[1, :], 'r*', alpha=0.5, label="Estimated Rotated Ellipse")
plt.legend()
plt.title("Comparison of Original and Estimated Rotated Ellipses")
plt.show()

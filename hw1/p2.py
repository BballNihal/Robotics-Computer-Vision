import numpy as np
from numpy.linalg import inv

# X and Y coordinates from problem description
x = [-1, 2.5, 6.0, -3.5, 3.0, 8.5]
y = [8.5, 3.0, -5.5, 13.0, 0.0, -10.0]

# TODO: Form the A matrix and the b vector
A = np.zeros((len(x), 2))
A[:, 0] = x
A[:, 1] = 1
b = np.array(y) 

# TODO: Find the vector q
q = inv(A.T @ A) @ A.T @ b   

# Display computed q vector
print('y = {0:2.3f}x + {1:2.3f}'.format(q[0], q[1]))

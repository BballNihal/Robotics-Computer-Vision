import numpy as np


matrix = np.array([[-1.32, -0.18, 2.13],
                   [2.64, -4.68, 4.65],
                   [1.47, -4.75, 6.80]])


rank = np.linalg.matrix_rank(matrix)

print("Matrix has exactly two independent columns:")
if rank == 2:
    print("true")
else:
    print("false")

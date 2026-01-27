import numpy as np

mat = np.random.randint(1, 100, (5, 5))
mean_val = mat.mean()

binary = np.where(mat > mean_val, 1, 0)

print("Original Matrix:\n", mat)
print("Binary Matrix based on Mean:\n", binary)

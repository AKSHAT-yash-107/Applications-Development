import numpy as np

# Step 1: Create random 10x10 matrix
matrix = np.random.randint(0, 10, (10, 10))
print("Matrix:\n", matrix)

# Step 2: Extract all 3x3 contiguous blocks
blocks = []

for i in range(10 - 3 + 1):
    for j in range(10 - 3 + 1):
        block = matrix[i:i+3, j:j+3]
        blocks.append(block)

# Example: print first block
print("\nFirst 3x3 block:\n", blocks[0])
print("\nTotal blocks:", len(blocks))
print("\n" )
print(blocks)


def is_magic(grid):
    arr=np.array(grid)
    b=arr.flatten()
    tar=sum(grid[0])
    if sorted(b)!=list(range(1,10)):
            return False

    #row sum
    for i in range(3):
        if sum(grid[i])!=tar:
            return False

    for i in range(3):
        for j in range(3):
            if grid[i][j]!=tar:
                return False

    for i in range(3):
        if sum(grid[i][i])!=tar:
            return False

    for i in range(3):
        if sum(grid[i][3-i-1])!=tar:
            return False

    return True

def count_blocks(grid):
    n=len(grid)
    count =0

    for i in range(n):
        for j in range(n):
            block=grid[i:i+3,j:j+3]
            if is_magic(block):
                count+=1

    return count
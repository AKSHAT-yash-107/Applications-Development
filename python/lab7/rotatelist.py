lst = [51, 10, 23, 44,452]
k = 2
n = len(lst)
print(lst)
# Rotate right
k %= n
rotated = lst[-k:] + lst[:-k]
print("Rotated List:", rotated)

# Longest increasing contiguous subsequence
max_seq = []
current = [rotated[0]]

for i in range(1, n):
    if rotated[i] > rotated[i-1]:
        current.append(rotated[i])
    else:
        if len(current) > len(max_seq):
            max_seq = current
        current = [rotated[i]]

if len(current) > len(max_seq):
    max_seq = current

print("Longest Increasing Subsequence:", max_seq)

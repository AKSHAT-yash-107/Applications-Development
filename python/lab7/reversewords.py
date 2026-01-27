import string

s = input("ENTER ")
words = s.split()
result = []

for word in words:
    clean = word.strip(string.punctuation)
    reversed_word = clean[::-1]
    result.append(reversed_word)

print(" ".join(result))

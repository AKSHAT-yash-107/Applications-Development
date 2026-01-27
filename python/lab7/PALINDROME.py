def is_palindrome(n):
    return str(n) == str(n)[::-1]

def is_armstrong(n):
    digits = list(map(int, str(n)))
    power = len(digits)
    return sum(d**power for d in digits) == n

n = int(input("ENTER N 6"))
result = []
both_count = 0

for i in range(1, n+1):
    p = is_palindrome(i)
    a = is_armstrong(i)
    if p or a:
        result.append(i)
    if p and a:
        both_count += 1

print("Palindrome or Armstrong Numbers:", *result)
print("Count satisfying both:", both_count)

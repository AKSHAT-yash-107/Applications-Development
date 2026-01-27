def factor_details(n):
    factors = []
    for i in range(1, n+1):
        if n % i == 0:
            factors.append(i)

    count = len(factors)
    total = sum(factors)
    perfect = (total - n) == n

    return count, total, perfect

n = int(input("ENTER N "))
count, total, perfect = factor_details(n)

print("Factors:", count)
print("Sum:", total)
print("Perfect Number:", "Yes" if perfect else "No")

paisa = int(input("paisa :"))

rupees = paisa // 100
remaining_paisa = paisa % 100

print(f"{rupees} Rupees and {remaining_paisa} Paisa")


possible = False
for i in range(rupees // 5 + 1):
    if (rupees - 5*i) % 10 == 0:
        possible = True
        break

print("Only 5 & 10 Rupee Notes Possible:", "Yes" if possible else "No")

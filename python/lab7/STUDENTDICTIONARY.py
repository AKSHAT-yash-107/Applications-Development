students = {'Amit': 80, 'Rina': 49, 'Suman': 70, 'Neha': 80}

avg = sum(students.values()) / len(students)

for name in students:
    if students[name] < avg:
        students[name] *= 1.05

print(students)

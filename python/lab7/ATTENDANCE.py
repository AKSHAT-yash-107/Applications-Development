import pandas as pd

data = {
    'Name': ['Amit', 'Rina', 'Suman'],
    'Attendance': [82, 68, 90]
}

df = pd.DataFrame(data)

df['Status'] = df['Attendance'].apply(
    lambda x: 'Eligible' if x >= 75 else 'Not Eligible'
)

eligible_count = (df['Status'] == 'Eligible').sum()

print(df)
print("Eligible Students Count:", eligible_count)

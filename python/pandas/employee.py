import pandas as pd
import numpy as np
df=pd.read_csv(r'emp.csv')


df.loc[df['Experience']>=5 , 'NEW SALARY' ]= df["Salary"]*1.1
df.loc[df['Experience']<5 , 'NEW SALARY' ]= df["Salary"]*1.05
print(df)


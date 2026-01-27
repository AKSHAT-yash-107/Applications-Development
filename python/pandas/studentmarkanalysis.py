import pandas as pd
import numpy as np
df=pd.read_csv(r'student.csv')

df['TOTAL MARKS']= df['Maths']+ df['Science']+df['English']
df['AVG MARKS']= (df['Maths']+ df['Science']+df['English'])/3
def result(x):
    return 'PASS' if x>=50 else 'FAIL'
df['RESULT']=df['AVG MARKS'].apply(result)


print(df)
import pandas as pd
import numpy as np
df=pd.read_csv(r'attendance_10 - attendance_10.csv')
print(df[df['Attendance']<75])
print(df[df['Attendance']<75].shape[0])

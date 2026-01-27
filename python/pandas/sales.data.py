import pandas as pd
import numpy as np
df=pd.read_csv(r'sales_data_10 - sales_data_10.csv')
print("total sales")
print(df.groupby('Product')["Sales"].sum())
print("avgsale")
print(df.groupby('Product')["Sales"].mean())
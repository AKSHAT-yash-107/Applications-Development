import pandas as pd

df = pd.read_csv(r"sales_data_10 - sales_data_10.csv")

df['Growth'] = df.groupby('Product')['Sales'].diff()

product = df.loc[df['Growth'].idxmax(), 'Product']

print("Highest Growth Product:", product)

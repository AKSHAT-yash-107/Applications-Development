import pandas as pd
import numpy as np
df=pd.read_csv(r"missing_marks_10 - missing_marks_10.csv")
print("missing" )
print(df[df["Marks"].isna()])
print("filled wih mean")
print(df.fillna({"Marks":df["Marks"].mean()}))



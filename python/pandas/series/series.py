import pandas as pd
import numpy as np
labels=['a','b','c']
mylist=[10,20,30]
arr=np.array([10,20,30])
d=pd.Series(mylist,index=labels)
print(d)
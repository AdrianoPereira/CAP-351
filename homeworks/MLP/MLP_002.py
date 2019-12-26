import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt


columns = ['class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 
           'deg-malig', 'breast', 'breast-quad', 'irradiat']

df = pd.read_csv('./data/breast-cancer.data', names=columns)

group = df.groupby(['class', 'age']).agg({'breast': 'count'})


width = 0.35
l1 = np.array(group.index.to_list())[:, 0]
l2 = np.array(group.index.to_list())[:, 1]

x  = np.arange(len(l2))
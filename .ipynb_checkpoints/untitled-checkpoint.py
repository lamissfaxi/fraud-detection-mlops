import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../data/creditcard.csv')
print(df.shape)
print(df['Class'].value_counts())
df.head()
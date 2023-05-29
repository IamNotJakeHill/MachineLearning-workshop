import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
df = pd.read_csv('GroceryStoreDataSet.csv',names=['Products'],header=None)

df.replace(to_replace='COCK', value='COKE', regex=True, inplace=True)
df.replace(to_replace='SUGER', value='SUGAR', regex=True, inplace=True)

data = list(df["Products"].apply(lambda x:x.split(',')))

a = TransactionEncoder()
a_data = a.fit(data).transform(data)
print(a_data)
df2 = pd.DataFrame(a_data,columns=a.columns_)

df3 = apriori(df2, min_support=0.01, use_colnames=True)
df_ar = association_rules(df3, metric="confidence", min_threshold=0.06)
print(df3.head(10))
print(df_ar.head(10))
dfc = df_ar.sort_values(by="confidence", ascending=False)
dfl = df_ar.sort_values(by="lift", ascending=False)
print(dfc.head(1))
print(dfl.head(1))
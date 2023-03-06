import numpy as np
import github
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pip._internal.vcs import git
from sklearn.preprocessing import LabelEncoder
import math
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split

#zaczytaj dane z pliku csv
df_train = pd.read_csv("train.csv" , sep = "," , encoding = 'utf-8', low_memory=False )
#sprawdź liczbę kolumn i wierszy
df_train.shape
df_train.info()
#wyświetl część tabeli
df_train.head()
#usuń wiersze z duplikatami id
df_train.drop_duplicates(subset="ID", inplace=True)

#opisz statystyki danych
df_train.describe()
#zlicz różne wartości danych
for i in df_train.columns:
    print(df_train[i].value_counts())
    print('*'*50)
    # pokaż liczności danych kategorycznych
sns.countplot(df_train['Credit_Score'])
#zastąp błędne dane

df_train.info()
#zmień dane na numeryczne
FeaturesToConvert = ['Age', 'Annual_Income',
'Num_of_Loan', 'Num_of_Delayed_Payment',
'Changed_Credit_Limit', 'Outstanding_Debt',
'Amount_invested_monthly', 'Monthly_Balance']

# ale najpierw sprawdź czy nie ma błędów w danych
for feature in FeaturesToConvert:
    uniques = df_train[feature].unique()
    print('Feature:', '\n', feature, '\n', uniques, '\n', '--'*40, '\n')

df_train.info()
#zmień dane na numeryczne
FeaturesToConvert = ['Age', 'Annual_Income',
'Num_of_Loan', 'Num_of_Delayed_Payment',
'Changed_Credit_Limit', 'Outstanding_Debt',
'Amount_invested_monthly', 'Monthly_Balance']
# ale najpierw sprawdź czy nie ma błędów w danych
for feature in FeaturesToConvert:
    uniques = df_train[feature].unique()
    print('Feature:', '\n', feature, '\n', uniques, '\n', '--'*40, '\n')
    # usuń zbędne znaki '-’ , '_'
    for feature in FeaturesToConvert:
        df_train[feature] = df_train[feature].str.strip('-_');
        # puste kolumny zastąp NAN
    for feature in FeaturesToConvert:
        df_train[feature] =df_train[feature].replace({'':np.nan});
        # zmien typ zmiennych ilościowych
    for feature in FeaturesToConvert:
        df_train[feature] = df_train[feature].astype('float64');

df_train['Monthly_Inhand_Salary']= df_train['Monthly_Inhand_Salary'].fillna(method='pad')

from sklearn.preprocessing import LabelEncoder
# stwórz obiekt enkodera
le = LabelEncoder()
df_train.Occupation = le.fit_transform(df_train.Occupation)
# sprawdź transformacje
df_train.head()


cols = ['workex', 'status', 'hsc_s', 'degree_t']
# Encode labels of multiple columns at once
df_train[cols] = df_train[cols].apply(LabelEncoder().fit_transform)
# Print head
df_train.head()

df_train['Credit_History_Age'].head()

plt.figure(figsize = (20, 18))
sns.heatmap(df_train['Credit_History_Age'].corr(), annot =True, linewidths = 0.1, cmap = 'Blues')
plt.title('Numerical Features Correlation')
plt.show()

scaler = MinMaxScaler()
col_float = ['Age', 'Annual_Income',
'Delay_from_due_date', 'Num_of_Delayed_Payment',
'Outstanding_Debt', 'Credit_History_Age',
'Total_EMI_per_month', 'Monthly_Balance']
df_cleaned = []
for i in col_float:
    df_cleaned[i] =scaler.fit_transform(df_cleaned[[i]])
    df_cleaned.head()

Q1 = df_cleaned.Annual_Income.quantile(0.25)
Q3 = df_cleaned.Annual_Income.quantile(0.75)
IQR = Q3 - Q1
df_cleaned =df_cleaned.drop(df_cleaned.loc[df_cleaned['Annual_Income'] > (Q3 + 1.5 * IQR)].index)
df_cleaned =df_cleaned.drop(df_cleaned.loc[df_cleaned['Annual_Income'] < (Q1 - 1.5 * IQR)].index)

from sklearn.model_selection import train_test_split
    # git.commit-m "przygotowanie danych"
    # git.push
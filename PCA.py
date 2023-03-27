import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# Zaczytanie danych z pliku
df = pd.read_csv("cleaned_data.csv", sep=",", encoding='utf-8', index_col= 0)
frame = df.drop(['Credit_Score'], axis = 1)
creditScore = df['Credit_Score']

# Wyświetl dane (początkowe)
print(frame.head())

# Cechy:
# sprawdzenie rozmiaru
print(frame.shape[1])

# sprawdzenie nazw kolumn i ich typów
print(frame.info())

# Przekształcenie zbioru:
pca = PCA(svd_solver= 'full', n_components= 0.90)
glowne = pca.fit_transform(frame)
glowneFrame = pd.DataFrame(data = glowne)

# Wyswietl dane
print(glowneFrame.head())

#Cechy:
# sprawdzenie rozmiaru
print(glowneFrame.shape[1])

# sprawdzenie nazw kolumn i ich typów
print(glowneFrame.info())

x_train, x_test, y_train, y_test = train_test_split(glowneFrame, creditScore, test_size= 0.5, random_state= 69)

print(x_train)
print(x_test)
print(y_train)
print(y_test)
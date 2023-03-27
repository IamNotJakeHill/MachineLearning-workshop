import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.linear_model import LogisticRegression
import matplotlib
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

x_train, x_test, y_train, y_test = train_test_split(glowneFrame, creditScore, test_size= 0.5, random_state= 100)

clf = LogisticRegression(random_state=100)
clf.fit(x_train, y_train)
clf.predict(x_test)

matrix = confusion_matrix(y_test,y_train)
ConfusionMatrixDisplay(matrix).plot()
plt.show()
raport = classification_report(y_test,y_train)
print(raport)

LogisticRegression(random_state=100)



print(x_train)
print(x_test)
print(y_train)
print(y_test)
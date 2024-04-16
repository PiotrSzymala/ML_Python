import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Wczytanie danych
iris_data = pd.read_csv('C:\\Users\\PS\\Pulpit\\Iris.csv')

# Sprawdzenie brakujących danych
missing_data = iris_data.isnull().sum()
print("Brakujące dane w każdej kolumnie:\n", missing_data)

# Sprawdzenie duplikatów
duplicates = iris_data.duplicated().sum()
print("Liczba duplikatów w zbiorze danych:", duplicates)
# Usuwanie duplikatów
if duplicates > 0:
    iris_data = iris_data.drop_duplicates()
    print("Duplikaty zostały usunięte. Nowa liczba rekordów:", iris_data.shape[0])

# Normalizacja i standaryzacja danych
# Usuwamy niepotrzebne kolumny przed skalowaniem
features = iris_data.drop(['Id', 'Species'], axis=1)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
print("Dane zostały znormalizowane i przeskalowane.")

# Dodanie sprawdzenia danych zeskalowanych
scaled_iris_data = pd.DataFrame(scaled_features, columns=features.columns)
scaled_iris_data['Species'] = iris_data['Species']
print("Pierwsze 5 wierszy danych zeskalowanych:\n", scaled_iris_data.head())

# Złączenie przeskalowanych danych z kolumną 'Species'
scaled_iris_data = pd.DataFrame(scaled_features, columns=features.columns)
scaled_iris_data['Species'] = iris_data['Species']

# Podział danych na zbiór uczący i testowy
X_train, X_test, y_train, y_test = train_test_split(
    scaled_iris_data.drop('Species', axis=1), scaled_iris_data['Species'], test_size=0.2, random_state=42)

# Wyświetlenie wyników podziału
print("Rozmiar zbioru uczącego:", X_train.shape[0])
print("Rozmiar zbioru testowego:", X_test.shape[0])

# Wnioski i obserwacje
print("Wnioski i obserwacje:")
print("1. Dane zostały sprawdzone pod kątem brakujących wartości i duplikatów. Brakujące wartości: {}, Duplikaty: {}."
      .format(missing_data.sum(), duplicates))
print("2. Dane zostały znormalizowane i standaryzowane, co jest kluczowe dla wielu algorytmów uczenia maszynowego.")
print("3. Zbiór danych został podzielony na uczący i testowy, co umożliwi ocenę modelu na niezobaczonych wcześniej danych.")

# Ustawienie stylu wykresów
sns.set(style="whitegrid")

# Histogramy dla każdej cechy
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))
sns.histplot(data=iris_data, x='SepalLengthCm', hue='Species', multiple='stack', ax=axes[0, 0])
axes[0, 0].set_title('Histogram of Sepal Lengths')
sns.histplot(data=iris_data, x='SepalWidthCm', hue='Species', multiple='stack', ax=axes[0, 1])
axes[0, 1].set_title('Histogram of Sepal Widths')
sns.histplot(data=iris_data, x='PetalLengthCm', hue='Species', multiple='stack', ax=axes[1, 0])
axes[1, 0].set_title('Histogram of Petal Lengths')
sns.histplot(data=iris_data, x='PetalWidthCm', hue='Species', multiple='stack', ax=axes[1, 1])
axes[1, 1].set_title('Histogram of Petal Widths')
plt.tight_layout()
plt.show()

# Wykresy pudełkowe dla każdej cechy
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))
sns.boxplot(x='Species', y='SepalLengthCm', data=iris_data, ax=axes[0, 0])
axes[0, 0].set_title('Boxplot of Sepal Lengths')
sns.boxplot(x='Species', y='SepalWidthCm', data=iris_data, ax=axes[0, 1])
axes[0, 1].set_title('Boxplot of Sepal Widths')
sns.boxplot(x='Species', y='PetalLengthCm', data=iris_data, ax=axes[1, 0])
axes[1, 0].set_title('Boxplot of Petal Lengths')
sns.boxplot(x='Species', y='PetalWidthCm', data=iris_data, ax=axes[1, 1])
axes[1, 1].set_title('Boxplot of Petal Widths')
plt.tight_layout()
plt.show()


# OBSERWACJE
# Normalizacja i standaryzacja danych jest istotna dla algorytmów uczenia maszynowego i prowadzi do lepszej interpretacji wyników.
# Wizualizacje są bardzo istotne do zrozumienia natury danych i ułatwiają identyfikację potencjalnych wzorców lub anomalii.

# WNIOSKI
# W uczeniu maszynowym wizualizacja jest nieodłącznym elementem operowania na danych i znacząco przyczynia się do odkrywania powtarzających się schematów czy też anomalii w zbiorze.
# By umożliwić bezproblemową wizualizacje danych konieczna jest najpierw normalizacja oraz standaryzacja zbioru.

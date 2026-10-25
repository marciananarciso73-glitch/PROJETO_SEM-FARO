import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

# 1. Carregar o dataset
df = pd.read_csv("data/dataset_farol.csv")

# 2. Verificar valores nulos
print("Valores nulos antes do tratamento:")
print(df.isnull().sum())

# 3. Tratar valores nulos preenchendo com a média
colunas = ["num_carros", "num_pedestres", "tempo_espera_pedestre_s"]
imputer = SimpleImputer(strategy="mean")
df[colunas] = imputer.fit_transform(df[colunas])

# 4. Normalizar os dados numéricos
scaler = MinMaxScaler()
df[colunas] = scaler.fit_transform(df[colunas])

print("Dados após o pré-processamento:")
print(df.head())

# 5. Separar features (X) e alvo (y)
X = df.drop("decisao_alvo", axis=1)
y = df["decisao_alvo"]

# 6. Dividir em treino e teste (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
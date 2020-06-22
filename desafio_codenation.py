import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Lendo os datasets de treino e teste e gerando um DataFrame de resposta
df_train = pd.read_csv(r"datasets\\train.csv", sep=",", encoding="utf-8")
df_test = pd.read_csv(r"datasets\\train.csv", sep=",", encoding="utf-8")
df_answer = pd.DataFrame()

df_answer["NU_INSCRICAO"] = df_test["NU_INSCRICAO"]

# Checando as correlações
correlation = df_train.corr()

features = correlation[
    (correlation["NU_NOTA_MT"] <= -0.3)
    | (correlation["NU_NOTA_MT"] >= 0.3) & (correlation["NU_NOTA_MT"] < 1.0)
]["NU_NOTA_MT"]

# Adicionando as notas de 'NU_NOTA_MT a lista de features para treinar o modelo
features = features.index.tolist()
features.append("NU_NOTA_MT")

# Função que preenche os valores NaN com 0 das colunas do Dataset


def fill_nan(dataset):
    for feature in features:
        dataset[feature].fillna(0, inplace=True)
    return dataset


df_train = df_train[features]
df_train = fill_nan(df_train)

# Removendo "NU_NOTA_MT" da lista de features para separar os dados para treinar o modelo
features.remove("NU_NOTA_MT")

# Selecionando as features para o Dataset de Teste
df_test = df_test[features]
df_test.fillna(0, inplace=True)

# Separando os dados de teste
x_test = df_test.copy()

# Separando os dados de treino
x_train = df_train.drop("NU_NOTA_MT", axis=1)
y_train = df_train["NU_NOTA_MT"]

# Gerando um Pipeline para treinar o modelo

rfr_pipe = Pipeline([("scaler", StandardScaler()), ("rfr", RandomForestRegressor())])

# Gerando uma lista com parametros para testes exaustivos com o GridSearch
val_estimator = [200, 250, 300]
val_criterion = ["mse"]
val_mx_features = ["log2"]

paramets = dict(
    rfr__n_estimators = val_estimator,
    rfr__criterion = val_criterion,
    rfr__max_features = val_mx_features,
)

# Instanciando o GridSearchCV
grid = GridSearchCV(rfr_pipe, paramets, cv=5, n_jobs=-1)

# Treinando o modelo
grid.fit(x_train, y_train)

pred = grid.predict(x_test)
df_answer["NU_NOTA_MT"] = pred
df_answer.to_csv("answer.csv", index=False, header=True)

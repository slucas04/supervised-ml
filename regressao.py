import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Regressão
# 1)
dataenzimas = np.loadtxt('atividade_enzimatica.csv', delimiter=',')

# Gráfico de atividade_enzimatica.csv
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(dataenzimas[:, 0], dataenzimas[:, 1], dataenzimas[:, 2])
ax.set_xlabel("Temperatura")
ax.set_ylabel("pH da solução")
ax.set_zlabel("Atividade enzimática")

plt.show()

# 2)
X = np.array(dataenzimas[:, 0:2])
y = np.array(dataenzimas[:, 2])
X1 = dataenzimas[:, 0]
X2 = dataenzimas[:, 1]

# 3)
# MQO tradicional
X = np.hstack((np.ones((X.shape[0], 1)), X))

beta_MQO = np.linalg.inv(X.T @ X) @ X.T @ y

print("Coeficientes: ", beta_MQO)

y_pred = X @ beta_MQO
RMSE = np.sqrt(np.mean((y - y_pred) ** 2))

print("Erro residual: ", RMSE)

# Gráfico MQO tradicional
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X1, X2, y, color='blue', label='Valores reais')
ax.scatter(X1, X2, y_pred, color='red', label='Valores preditos com MQO')
ax.set_xlabel("Temperatura")
ax.set_ylabel("pH da solução")
ax.set_zlabel("Atividade enzimática")

ax.legend()

plt.show()

# 4)
# MQO regularizado
lambdas = [0, 0.25, 0.5, 0.75, 1]
betas = []

for lambda_ in lambdas:
    I = np.eye(X.shape[1])
    I[0, 0] = 0 
    beta_MQO_regularizado = np.linalg.inv(X.T @ X + lambda_ * I) @ X.T @ y
    betas.append(beta_MQO_regularizado)

for i, lambda_ in enumerate(lambdas):
    print(f"Coeficientes para λ = {lambda_}:")
    print(betas[i])
    print()

# Média de valores observáveis
y_media = np.mean(y)
y_pred_media = np.full_like(y, y_media)

# Gráfico de média de valores observáveis
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X1, X2, y, color='blue', label='Valores reais')
ax.scatter(X1, X2, y_pred_media, color='red', label='Valores preditos com média')
ax.set_xlabel("Temperatura")
ax.set_ylabel("pH da solução")
ax.set_zlabel("Atividade enzimática")

ax.legend()

plt.show()

# 5)
# Validação via Monte Carlo
rss_MQO = []
rss_MQO_regularizado = [[], [], [], [], []]
rss_media = []


for i in range(500):
    indices = np.random.permutation(X.shape[0])
    n_treino = int(0.8 * X.shape[0])
    treino_idx = indices[:n_treino]
    teste_idx = indices[n_treino:]
    X_treino = X[treino_idx, :]
    y_treino = y[treino_idx]
    X_teste = X[teste_idx, :]
    y_teste = y[teste_idx]

    # MQO tradicional
    beta = np.linalg.inv(X_treino.T @ X_treino) @ X_treino.T @ y_treino
    y_pred = X_teste @ beta
    rss = np.sum((y_teste - y_pred) ** 2)
    rss_MQO.append(rss)

    # MQO regularizado
    for lambda_ in lambdas:
        I = np.eye(X_treino.shape[1])
        I[0, 0] = 0
        beta_regularizado = np.linalg.inv(X_treino.T @ X_treino + lambda_ * I) @ X_treino.T @ y_treino
        y_pred_regularizado = X_teste @ beta_regularizado
        rss_regularizado = np.sum((y_teste - y_pred_regularizado) ** 2)
        rss_MQO_regularizado[lambdas.index(lambda_)].append(rss_regularizado)

    # Média de valores observáveis
    y_media_treino = np.mean(y_treino)
    y_pred_media = np.full_like(y_teste, y_media_treino)
    rss_med = np.sum((y_teste - y_pred_media) ** 2)
    rss_media.append(rss_med)


media_MQO = np.mean(rss_MQO)
media_MQO_regularizado = [np.mean(rss) for rss in rss_MQO_regularizado]
media_media = np.mean(rss_media)

desvio_MQO = np.std(rss_MQO)
desvio_MQO_regularizado = [np.std(rss) for rss in rss_MQO_regularizado]
desvio_media = np.std(rss_media)

max_MQO = np.max(rss_MQO)
max_MQO_regularizado = [np.max(rss) for rss in rss_MQO_regularizado]
max_media = np.max(rss_media)

min_MQO = np.min(rss_MQO)
min_MQO_regularizado = [np.min(rss) for rss in rss_MQO_regularizado]
min_media = np.min(rss_media)

print("Resultados:")
print(media_MQO)
print(media_MQO_regularizado)
print(media_media)


modelos = ["MQO"] + [f"Reg λ={l}" for l in [0, 0.25, 0.5, 0.75, 1]] + ["Média"]
metricas = ["Média", "Desvio Padrão", "Máximo", "Mínimo"]
dados = np.array([
    [media_MQO] + media_MQO_regularizado + [media_media],
    [desvio_MQO] + desvio_MQO_regularizado + [desvio_media],
    [max_MQO] + max_MQO_regularizado + [max_media],
    [min_MQO] + min_MQO_regularizado + [min_media]
]).T
fig, ax = plt.subplots(figsize=(8, len(modelos) * 0.5))
ax.axis('off')

tabela = ax.table(cellText=dados, 
                  rowLabels=modelos, 
                  colLabels=metricas, 
                  cellLoc='center', 
                  loc='center')

tabela.auto_set_font_size(False)
tabela.set_fontsize(8)
tabela.auto_set_column_width([i for i in range(len(metricas))])
tabela.scale(1, 1)

plt.show()

plt.figure(figsize=(8, 6))
plt.errorbar(lambdas, media_MQO_regularizado, yerr=desvio_MQO_regularizado, fmt='o-', capsize=5, label='Tikhonov')
plt.xlabel('λ')
plt.ylabel('RSS Médio')
plt.title('Comparação dos Resultados de Tikhonov para Diferentes λ')
plt.grid(True)
plt.legend()
plt.show()
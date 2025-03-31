import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors

# Classificação
# 1)
dataexpressoes = np.loadtxt('EMGsDataset.csv', delimiter=',')
X = dataexpressoes[:2, :]
y = dataexpressoes[2, :]
y_labels = y.copy()
classes = np.unique(y_labels)
C = len(classes)
N = y_labels.shape[0]

Y_onehot = np.zeros((N, C))
for i, cls in enumerate(classes):
    Y_onehot[y_labels == cls, i] = 1

Y_gaussian = Y_onehot.T

X1 = X[0, :]
X2 = X[1, :]

# 2)
colors = plt.cm.get_cmap('tab10', len(classes))
norm = mcolors.BoundaryNorm(classes, colors.N)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i, classe in enumerate(classes):
    mask = y_labels == classe
    ax.scatter(X1[mask], X2[mask], y_labels[mask], color=colors(i), label=f"Classe {classe}")

ax.set_xlabel("Corrugador do Supercílio")
ax.set_ylabel("Zigomático Maior")
ax.set_zlabel("Expressão")
ax.legend(title="Classes", loc='upper right')
plt.show()

# 3)
# MQO tradicional
X_MQO = X.T
y_MQO = y.T
X1_MQO = X_MQO[:, 0]
X2_MQO = X_MQO[:, 1]

X_MQO = np.hstack((np.ones((X_MQO.shape[0], 1)), X_MQO))

beta_MQO = np.linalg.inv(X_MQO.T @ X_MQO) @ X_MQO.T @ y_MQO

print("Coeficientes: ", beta_MQO)

y_pred_MQO = X_MQO @ beta_MQO
RMSE = np.sqrt(np.mean((y_MQO - y_pred_MQO) ** 2))

print("Erro residual: ", RMSE)

# Gráfico MQO tradicional
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(X1_MQO, X2_MQO, y_MQO, color='blue', label='Valores reais')

ax.scatter(X1_MQO, X2_MQO, y_pred_MQO, color='red', label='Valores preditos MQO')
ax.set_xlabel("Corrugador do Supercílio")
ax.set_ylabel("Zigomático Maior")
ax.set_zlabel("Expressão")

ax.legend()

plt.show()

# Classificador Gaussiano Tradicional
parametros = {}

for c in classes:
    X_c = X[:, y == c]
    media_c = np.mean(X_c, axis=0)
    cov_c = np.cov(X_c)
    parametros[c] = (media_c, cov_c)

priors = {}
n_total = len(y)
for c in classes:
    n_c = np.sum(y == c)
    priors[c] = n_c / n_total

def FDP(x, media, cov):
    d = len(media)
    diff = x - media
    # Adiciona uma pequena constante na diagonal para regularização
    cov_reg = cov + 1e-6 * np.eye(d)
    inv_cov = np.linalg.inv(cov_reg)
    det_cov = np.linalg.det(cov_reg)
    denom = np.power(2 * np.pi, d / 2) * np.sqrt(det_cov)
    exponent = -0.5 * (diff.T @ inv_cov @ diff)
    return np.exp(exponent) / denom


def classificar(x_novo, parametros, priors):
    probabilidades = {}
    for c in parametros:
        media_c, cov_c = parametros[c]
        likelihood = FDP(x_novo, media_c, cov_c)
        probabilidades[c] = likelihood * priors[c]
    return max(probabilidades, key=probabilidades.get), probabilidades

# Classificador Gaussiano Com Covariâncias Iguais
parametros = {}

for c in classes:
    X_c = X[:, y == c]
    media_c = np.mean(X_c, axis=0)
    cov_c = np.cov(X)
    parametros[c] = (media_c, cov_c)

# Classificador Gaussiano com Matriz Agregada
parametros = {}
cov_sum = None
total_samples = 50000

for c in classes:
    X_c = X[:, y == c]
    media_c = np.mean(X_c, axis=0)
    cov_c = np.cov(X_c)
    parametros[c] = media_c
    n_c = np.sum(y == c)
    if cov_sum is None:
        cov_sum = (n_c - 1) * cov_c
    else:
        cov_sum += (n_c - 1) * cov_c

K = 5
cov_pooled = cov_sum / (total_samples - K)

def classificar_agregado(x_novo, parametros, priors):
    probabilidades = {}
    for c in parametros:
        media_c = parametros[c]
        likelihood = FDP(x_novo, media_c, cov_pooled)
        probabilidades[c] = likelihood * priors[c]
    return max(probabilidades, key=probabilidades.get), probabilidades

# Classificador Gaussiano Regularizado (Friedman)
lambdas = [0, 0.25, 0.5, 0.75, 1]
desempenho_lambda = {}

def regularize_cov(cov, lambda_reg):
    T = np.diag(np.diag(cov))
    return (1 - lambda_reg) * cov + lambda_reg * T

def FDP_reg(x, media, cov_reg):
    d = len(media)
    diff = x - media
    inv_cov = np.linalg.inv(cov_reg)
    det_cov = np.linalg.det(cov_reg)
    denom = np.power(2 * np.pi, d/2) * np.sqrt(det_cov)
    exponent = -0.5 * (diff.T @ inv_cov @ diff)
    return np.exp(exponent) / denom

def classificar_regularizado(x_novo, parametros, priors, cov_reg):
    probabilidades = {}
    for c in parametros:
        media_c = parametros[c]
        likelihood = FDP_reg(x_novo, media_c, cov_reg)
        probabilidades[c] = likelihood * priors[c]
    return max(probabilidades, key=probabilidades.get)
    
# Classificador de Bayes Ingênuo
parametros = {}
priors = {}
n_total = len(y)

for c in classes:
    X_c = X[:, y == c]
    media_c = np.mean(X_c, axis=1)
    var_c = np.var(X_c, axis=1)
    parametros[c] = (media_c, var_c)
    n_c = X_c.shape[1]
    priors[c] = n_c / n_total

def gaussian_pdf(x, mu, var):
    epsilon = 1e-6  # Pequena constante para evitar divisão por zero
    safe_var = var + epsilon
    return (1.0 / np.sqrt(2 * np.pi * safe_var)) * np.exp(- (x - mu)**2 / (2 * safe_var))

def classificar_bayes_ingenuo(x, parametros, priors):
    probabilidades = {}
    for c in parametros:
        mu, var = parametros[c]
        likelihood = np.prod([gaussian_pdf(x[j], mu[j], var[j]) for j in range(len(x))])
        probabilidades[c] = likelihood * priors[c]
    return max(probabilidades, key=probabilidades.get), probabilidades


# 5)
# Monte Carlo

R = 500
acc_MQO = []
acc_gauss_trad = []
acc_gauss_cov_igual = []
acc_gauss_agregado = []
acc_gauss_reg = [[] for _ in lambdas]
acc_bayes_ing = []

for i in range(R):
    print(i)
    indices = np.random.permutation(X.shape[1])
    n_train = int(0.8 * X.shape[1])
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    X_train = X[:, train_idx]
    y_train = y[train_idx]
    X_test  = X[:, test_idx]
    y_test  = y[test_idx]
    
    # MQO Tradicional
    X_train_MQO = X_train.T
    X_train_MQO = np.hstack((np.ones((X_train_MQO.shape[0], 1)), X_train_MQO))
    beta = np.linalg.inv(X_train_MQO.T @ X_train_MQO) @ X_train_MQO.T @ y_train
    
    X_test_MQO = X_test.T
    X_test_MQO = np.hstack((np.ones((X_test_MQO.shape[0], 1)), X_test_MQO))
    y_pred = X_test_MQO @ beta
    y_pred_class = np.clip(np.rint(y_pred), classes.min(), classes.max())
    acc_MQO.append(np.mean(y_pred_class == y_test))
    
    # Classificador Gaussiano Tradicional
    params_trad = {}
    priors = {}
    for c in classes:
        X_c = X_train[:, y_train == c]
        params_trad[c] = (np.mean(X_c, axis=1), np.cov(X_c))
        priors[c] = X_c.shape[1] / X_train.shape[1]
        
    correct = 0
    for j in range(X_test.shape[1]):
        x_new = X_test[:, j]
        pred, _ = classificar(x_new, params_trad, priors)
        if pred == y_test[j]:
            correct += 1
    acc_gauss_trad.append(correct / X_test.shape[1])
    
    # Classificador Gaussiano com Covariâncias Iguais
    params_cov_igual = {}
    cov_equal = np.cov(X_train)
    for c in classes:
        X_c = X_train[:, y_train == c]
        params_cov_igual[c] = (np.mean(X_c, axis=1), cov_equal)
    correct = 0
    for j in range(X_test.shape[1]):
        x_new = X_test[:, j]
        pred, _ = classificar(x_new, params_cov_igual, priors)
        if pred == y_test[j]:
            correct += 1
    acc_gauss_cov_igual.append(correct / X_test.shape[1])
    
    # Classificador Gaussiano com Matriz Agregada
    params_agregado = {}
    cov_sum = None
    total_train = X_train.shape[1]
    for c in classes:
        X_c = X_train[:, y_train == c]
        params_agregado[c] = np.mean(X_c, axis=1)
        n_c = X_c.shape[1]
        if cov_sum is None:
            cov_sum = (n_c - 1) * np.cov(X_c)
        else:
            cov_sum += (n_c - 1) * np.cov(X_c)
    cov_pooled = cov_sum / (total_train - len(classes))
    correct = 0
    for j in range(X_test.shape[1]):
        x_new = X_test[:, j]
        prob = {}
        for c in params_agregado:
            likelihood = FDP(x_new, params_agregado[c], cov_pooled)
            prob[c] = likelihood * priors[c]
        pred = max(prob, key=prob.get)
        if pred == y_test[j]:
            correct += 1
    acc_gauss_agregado.append(correct / X_test.shape[1])
    
    # Classificador Gaussiano Regularizado (Friedman)
    for idx, lambda_ in enumerate(lambdas):
        cov_reg = regularize_cov(cov_pooled, lambda_)
        correct = 0
        for j in range(X_test.shape[1]):
            x_new = X_test[:, j]
            prob = {}
            for c in params_agregado:
                likelihood = FDP(x_new, params_agregado[c], cov_reg)
                prob[c] = likelihood * priors[c]
            pred = max(prob, key=prob.get)
            if pred == y_test[j]:
                correct += 1
        acc_gauss_reg[idx].append(correct / X_test.shape[1])
    
    # Classificador de Bayes Ingênuo
    params_bayes = {}
    for c in classes:
        X_c = X_train[:, y_train == c]
        params_bayes[c] = (np.mean(X_c, axis=1), np.var(X_c, axis=1))
    correct = 0
    for j in range(X_test.shape[1]):
        x_new = X_test[:, j]
        pred, _ = classificar_bayes_ingenuo(x_new, params_bayes, priors)
        if pred == y_test[j]:
            correct += 1
    acc_bayes_ing.append(correct / X_test.shape[1])

# 6)
# Cálculo das acurácias médias para cada modelo
avg_acc_MQO = np.mean(acc_MQO)
avg_acc_gauss_trad = np.mean(acc_gauss_trad)
avg_acc_gauss_cov_igual = np.mean(acc_gauss_cov_igual)
avg_acc_gauss_agregado = np.mean(acc_gauss_agregado)
avg_acc_gauss_reg = [np.mean(acc_list) for acc_list in acc_gauss_reg]
avg_acc_bayes_ing = np.mean(acc_bayes_ing)

print("Acurácias Médias:")
print("MQO Tradicional:", avg_acc_MQO)
print("Gaussiano Tradicional:", avg_acc_gauss_trad)
print("Gaussiano com Covariâncias Iguais:", avg_acc_gauss_cov_igual)
print("Gaussiano com Matriz Agregada:", avg_acc_gauss_agregado)
print("Gaussiano Regularizado (por λ):", avg_acc_gauss_reg)
print("Bayes Ingênuo:", avg_acc_bayes_ing)

def calcular_estatisticas(acuracias):
    return {
        "Média": np.mean(acuracias),
        "Desvio-padrão": np.std(acuracias, ddof=1),  # ddof=1 para amostra
        "Maior": np.max(acuracias),
        "Menor": np.min(acuracias)
    }

stats_MQO = calcular_estatisticas(acc_MQO)
stats_gauss_trad = calcular_estatisticas(acc_gauss_trad)
stats_gauss_cov_igual = calcular_estatisticas(acc_gauss_cov_igual)
stats_gauss_agregado = calcular_estatisticas(acc_gauss_agregado)
stats_gauss_reg = [calcular_estatisticas(acc_list) for acc_list in acc_gauss_reg]
stats_bayes_ing = calcular_estatisticas(acc_bayes_ing)

print("\nEstatísticas das Acurácias:")
print("MQO Tradicional:", stats_MQO)
print("Gaussiano Tradicional:", stats_gauss_trad)
print("Gaussiano com Covariâncias Iguais:", stats_gauss_cov_igual)
print("Gaussiano com Matriz Agregada:", stats_gauss_agregado)
print("Gaussiano Regularizado (por λ):", stats_gauss_reg)
print("Bayes Ingênuo:", stats_bayes_ing)
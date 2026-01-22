import numpy as np
from IPython.core.display import Latex
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
# ---------------------------
# 1. Generate synthetic data
# ---------------------------

N = 2000
Z = np.random.randn(N, 1)
X = 2 * Z + 0.5 * np.random.randn(N, 1)       # X depends on Z
Y = 3 * X + 4 * Z + np.random.randn(N, 1)     # true causal effect of X = 3

plt.figure()
plt.plot(X,Z, marker="o", linestyle="", markersize=1)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.show()
# ---------------------------
# 2. Naive observational model: Y ~ X
# ---------------------------

naive_model = LinearRegression()
naive_model.fit(X, Y)

print("Naive (biased) slope:", naive_model.coef_[0, 0])
# Incorrect because Z confounds X -> Y


# ---------------------------
# 3. Causal model: Y ~ X + Z
# ---------------------------

XZ = np.hstack([X, Z])

causal_model = LinearRegression()
causal_model.fit(XZ, Y)

w_X = causal_model.coef_[0, 0]
w_Z = causal_model.coef_[0, 1]

print("Causal w_X (true causal effect):", w_X)
print("Causal w_Z:", w_Z)
# w_X should be ~3, w_Z should be ~4


# ---------------------------
# 4. Interventional prediction: E[Y | do(X=x)]
# ---------------------------

x_test = np.array([[1.0], [2.0], [3.0]])

# Under do(X=x), we break Z → X and set Z = 0
XZ_do = np.hstack([x_test, np.zeros_like(x_test)])

Y_do = causal_model.predict(XZ_do)

print("E[Y | do(X)]:")
print(Y_do)


#%%
# Causal effect = derivative wrt X under do(X)
causal_effect = w_X
print("Causal effect d/dX E[Y | do(X)]:", causal_effect)
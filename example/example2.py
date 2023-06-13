import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import BayesianRidge


# A more sophisticated example than example1.py. In this example density from multiple simulations is used as a function of temperature.
# Density can be stored and retreived in the same way as example1.py. GPR is then used on simulation data.

x = [273.15, 283.15, 293.15, 303.15, 313.15, 323.15, 333.15, 343.15, 353.15, 363.15, 373.15]
y1 = [1.0200667, 1.0056195, 0.9901076, 0.9926550, 1.0139858,1.0011969, 0.9759674, 0.9611737, 0.9411064, 0.92950789, 0.9513683]
y2 = [0.99982, 0.9997, 0.99829, 0.99571, 0.99225, 0.98802, 0.98313, 0.97763, 0.9716, 0.9650, 0.9580]


plt.scatter(x, y1, label='Simulation')
plt.scatter(x, y2, label='Experiment')
# plt.xticks(x)
plt.xticks(rotation=45)
plt.xlabel('Temperature/ K')
plt.ylabel('Density/ gcm$^{-3}$')
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import BayesianRidge

x = np.array([273.15, 283.15, 293.15, 303.15, 313.15, 323.15, 333.15, 343.15, 353.15, 363.15, 373.15])
y = np.array([-0.0202467, -0.0059195, 0.0081824, 0.003055, -0.0217358, -0.0131769, 0.0071626, 0.0164563, 0.0304936, 0.03549211, 0.0066317])

X = np.vander(x, 4, increasing=True)

reg = BayesianRidge(tol=1e-6, fit_intercept=False, compute_score=True)
reg.fit(X, y)

x_pred = np.linspace(273.15, 373.15, 100)
X_pred = np.vander(x_pred, 4, increasing=True)
y_pred, y_std = reg.predict(X_pred, return_std=True)

plt.scatter(x, y, label='Observations')
plt.plot(x_pred, y_pred, color='red', label='Regression')
plt.fill_between(x_pred, y_pred - y_std, y_pred + y_std, color='pink', alpha=0.5, label='Uncertainty')
plt.xlabel('Temperature / K')
plt.ylabel('Y')
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

x = np.array([273.15, 283.15, 293.15, 303.15, 313.15, 323.15, 333.15, 343.15, 353.15, 363.15, 373.15])
y = np.array([-0.0202467, -0.0059195, 0.0081824, 0.003055, -0.0217358, -0.0131769, 0.0071626, 0.0164563, 0.0304936, 0.03549211, 0.0066317])

X = x[:, np.newaxis]  # Reshape x to 2D array

# Randomly select 6 training samples
rng = np.random.RandomState(1)
training_indices = rng.choice(np.arange(y.size), size=6, replace=False)
X_train, y_train = X[training_indices], y[training_indices]

kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
gaussian_process.fit(X_train, y_train)

x_pred = np.linspace(273.15, 373.15, 100)
X_pred = x_pred[:, np.newaxis]  # Reshape x_pred to 2D array
mean_prediction, std_prediction = gaussian_process.predict(X_pred, return_std=True)

plt.plot(x, y, label=r"$f(x)$", linestyle="dotted")
plt.scatter(X_train, y_train, label="Observations")
plt.plot(x_pred, mean_prediction, label="Mean prediction")
plt.fill_between(
    x_pred,
    mean_prediction - 1.96 * std_prediction,
    mean_prediction + 1.96 * std_prediction,
    alpha=0.5,
    label=r"95% confidence interval",
)
plt.legend()
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
plt.title("Gaussian process regression")
plt.show()
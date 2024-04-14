import numpy as np
from matplotlib import pyplot as plt
from skltemplate import TemplateEstimator

# Generating non-random data with a sinusoidal pattern
X = np.linspace(0, 10, 100)[:, np.newaxis]
y = np.sin(X).ravel() + np.random.normal(0, 0.1, size=X.shape[0])

# Fitting the template estimator
estimator = TemplateEstimator()
estimator.fit(X, y)

# Plotting with some fun elements
plt.figure(figsize=(8, 6))
plt.plot(X, estimator.predict(X), color='skyblue', linestyle='-', linewidth=2, label='Predicted')
plt.scatter(X, y, color='orange', marker='o', label='Actual', alpha=0.7)
plt.title("Sinusoidal Prediction", fontsize=16, fontweight='bold', color='purple')
plt.xlabel("X", fontsize=12, fontstyle='italic', color='green')
plt.ylabel("y", fontsize=12, fontstyle='italic', color='green')
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

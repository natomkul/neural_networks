import numpy as np
import matplotlib.pyplot as plt

# Dane XOR
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Funkcje aktywacji
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)

# Inicjalizacja wag
np.random.seed(42)
W1 = np.random.randn(2, 2)   # warstwa wejściowa -> ukryta
b1 = np.zeros((1, 2))
W2 = np.random.randn(2, 1)   # warstwa ukryta -> wyjście
b2 = np.zeros((1, 1))

lr = 0.1  # współczynnik uczenia
epochs = 10000

# HISTORIA
# - MSE
mse_output_history = []
mse_hidden_history = []

# - Błąd klasyfikacji
classification_error_history = []

# - Wagi
W1_history = []
W2_history = []

# Trening
for _ in range(epochs):
    # Zapis wag
    W1_history.append(W1.copy())
    W2_history.append(W2.copy())

    # Propagacja w przód
    z1 = sigmoid(X @ W1 + b1)
    y_hat = sigmoid(z1 @ W2 + b2)

    # MSE
    error = y - y_hat
    mse_output = np.mean(error ** 2)
    mse_output_history.append(mse_output)

    # Klasyfikacja 0–1
    y_pred = (y_hat >= 0.5).astype(int)
    class_error = np.mean(y_pred != y)
    classification_error_history.append(class_error)

    # Propagacja wsteczna
    d2 = error * sigmoid_deriv(y_hat)
    d1 = d2 @ W2.T * sigmoid_deriv(z1)

    # "MSE" w warstwie ukrytej (energia sygnału błędu)
    mse_hidden = np.mean(d1 ** 2)
    mse_hidden_history.append(mse_hidden)

    # Aktualizacja wag
    W2 += z1.T @ d2 * lr
    b2 += np.sum(d2, axis=0, keepdims=True) * lr
    W1 += X.T @ d1 * lr
    b1 += np.sum(d1, axis=0, keepdims=True) * lr

# Wykresy
# - MSE
plt.figure()
plt.plot(mse_output_history)
plt.title("MSE – warstwa wyjściowa (zbiór uczący)")
plt.xlabel("Epoka")
plt.ylabel("MSE")
plt.grid()

plt.figure()
plt.plot(mse_hidden_history)
plt.title("MSE sygnału błędu – warstwa ukryta")
plt.xlabel("Epoka")
plt.ylabel("MSE")
plt.grid()
 
# - Błąd klasyfikacji
plt.figure()
plt.plot(classification_error_history)
plt.title("Błąd klasyfikacji 0–1 (próg 0.5)")
plt.xlabel("Epoka")
plt.ylabel("Błąd klasyfikacji")
plt.ylim(0, 1)
plt.grid()

# -Wagi
# Konwersja do tablic
W1_history = np.array(W1_history)  # (epochs, 2, 2)
W2_history = np.array(W2_history)  # (epochs, 2, 1)

# -- Wykresy wag – warstwa 1
plt.figure()
plt.plot(W1_history[:, 0, 0], label="w11")
plt.plot(W1_history[:, 0, 1], label="w12")
plt.plot(W1_history[:, 1, 0], label="w21")
plt.plot(W1_history[:, 1, 1], label="w22")
plt.title("Wagi – warstwa wejściowa → ukryta")
plt.xlabel("Epoka")
plt.ylabel("Wartość wagi")
plt.legend()
plt.grid()

# -- Wykresy wag – warstwa 2
plt.figure()
plt.plot(W2_history[:, 0, 0], label="v1")
plt.plot(W2_history[:, 1, 0], label="v2")
plt.title("Wagi – warstwa ukryta → wyjście")
plt.xlabel("Epoka")
plt.ylabel("Wartość wagi")
plt.legend()
plt.grid()

plt.show()

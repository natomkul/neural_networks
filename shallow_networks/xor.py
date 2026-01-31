import numpy as np
import matplotlib.pyplot as plt

# Funkcje aktywacji
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Dane problemu XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Inicjalizacja sieci
np.random.seed(42)
input_dim = 2
hidden_dim = 2
output_dim = 1

# Wagi i biasy
wh = np.random.uniform(size=(input_dim, hidden_dim))
bh = np.random.uniform(size=(1, hidden_dim))
wout = np.random.uniform(size=(hidden_dim, output_dim))
bout = np.random.uniform(size=(1, output_dim))

# Parametry uczenia
lr = 0.5
epochs = 10000
target_mse = 0.001

# Historia błędów do wykresów
mse_total_history = []
mse_examples_history = []
class_error_history = []
wh_history = []
wout_history = []

# Pętla uczenia
for epoch in range(epochs):
    # --- Forward Propagation ---
    hidden_input = np.dot(X, wh) + bh
    hidden_output = sigmoid(hidden_input)
    
    final_input = np.dot(hidden_output, wout) + bout
    predicted = sigmoid(final_input)

    # OBLICZANIE MSE
    # każdego z osobna
    individual_mse = np.square(y - predicted)
    mse_examples_history.append(individual_mse.flatten())

    # całego zbioru
    mse_total = np.mean(individual_mse)
    mse_total_history.append(mse_total)

    # Błąd Klasyfikacji
    predictions_binary = (predicted >= 0.5).astype(int)
    error_rate = np.mean(predictions_binary != y)
    class_error_history.append(error_rate)

    # Zapisywanie wag
    wh_history.append(wh.copy())
    wout_history.append(wout.copy())

    # --- Backpropagation ---
    output_error = y - predicted
    d_predicted = output_error * sigmoid_derivative(predicted)
    
    hidden_error = d_predicted.dot(wout.T)
    d_hidden = hidden_error * sigmoid_derivative(hidden_output)

    # --- Aktualizacja wag i biasów ---
    wout += hidden_output.T.dot(d_predicted) * lr
    bout += np.sum(d_predicted, axis=0, keepdims=True) * lr
    wh += X.T.dot(d_hidden) * lr
    bh += np.sum(d_hidden, axis=0, keepdims=True) * lr

    # --- WARUNEK WCZESNEGO ZATRZYMANIA ---
    if mse_total <= target_mse:
        break



#  Wizualizacja wyników

mse_examples_history = np.array(mse_examples_history)
wh_history = np.array(wh_history)
wout_history = np.array(wout_history)

# Wykres 1: MSE dla każdego przykładu uczącego
plt.figure(figsize=(10, 5))
labels = ['XOR(0,0)', 'XOR(0,1)', 'XOR(1,0)', 'XOR(1,1)']
for i in range(4):
    plt.plot(mse_examples_history[:, i], label=labels[i])
plt.title('MSE dla poszczególnych przykładów uczących')
plt.xlabel('Epoki')
plt.ylabel('Kwadrat błędu')
plt.grid(True, alpha=0.3)
plt.legend()

# Wykres 2: MSE dla całego zbioru
plt.figure(figsize=(10, 5))
plt.plot(mse_total_history, color='black', linewidth=2, label='Total MSE')
plt.axhline(y=target_mse, color='red', linestyle='--', label='Cel')
plt.title('MSE dla całego zbioru uczącego')
plt.xlabel('Epoki')
plt.ylabel('Średni błąd kwadratowy')
plt.grid(True, alpha=0.3)
plt.legend()

# Wykres 3: Błąd klasyfikacji
plt.figure(figsize=(10, 7))
plt.plot(class_error_history, color='blue', linewidth=2, label='Błąd Klasyfikacji')
plt.title('Błąd Klasyfikacji (Próg 0.5)')
plt.xlabel('Epoki')
plt.ylabel('Błąd (0-1)')
plt.yticks([0, 0.25, 0.5, 0.75, 1.0])
plt.grid(True, alpha=0.3)
plt.legend()

# 5: Wykres Wag Warstwy Ukrytej (wh)
plt.figure(figsize=(10, 7))
for i in range(input_dim):
    for j in range(hidden_dim):
        plt.plot(wh_history[:, i, j], label=f'Waga wej{i+1}->ukr{j+1}')
plt.title('Ewolucja Wag: Warstwa Wejściowa -> Ukryta')
plt.xlabel('Epoki')
plt.legend(fontsize='small')
plt.grid(True, alpha=0.3)

# 4: Wykres Wag Warstwy Wyjściowej (wout)
plt.figure(figsize=(10, 7))
for i in range(hidden_dim):
    for j in range(output_dim):
        plt.plot(wout_history[:, i, j], label=f'Waga ukr{i+1}->wyj{j+1}')
plt.title('Ewolucja Wag: Warstwa Ukryta -> Wyjściowa')
plt.xlabel('Epoki')
plt.legend(fontsize='small')
plt.grid(True, alpha=0.3)


plt.tight_layout()
plt.show()


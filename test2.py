import numpy as np
import matplotlib.pyplot as plt

def linear_regression(x, w, b):
    return x * w + b

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def gradient_descent(x, y_true, w, b, learning_rate):
    y_pred = linear_regression(x, w, b)
    dw = -2 * np.mean(x * (y_true - y_pred))
    db = -2 * np.mean(y_true - y_pred)
    w -= learning_rate * dw
    b -= learning_rate * db
    return w, b

c = np.array([-40, -10, 0, 8, 15, 22, 38])
f = np.array([-40, 14, 32, 46, 59, 72, 100])

# Инициализация весов
w = np.random.rand()
b = np.random.rand()

learning_rate = 0.01
epochs = 500
loss_history = []

for epoch in range(epochs):
    w, b = gradient_descent(c, f, w, b, learning_rate)
    loss = mean_squared_error(f, linear_regression(c, w, b))
    loss_history.append(loss)

print("Обучение завершено")

# Вывод результатов
print(linear_regression(100, w, b))
print("Веса (w, b):", w, b)

# График функции потерь
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

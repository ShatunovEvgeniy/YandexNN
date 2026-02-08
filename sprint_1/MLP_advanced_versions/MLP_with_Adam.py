import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

class MLP:
    def __init__(self, layer_sizes, activation='tanh'):
        self.eps = 1e-8  # Маленькая константа для избежания деления на ноль
        self.weight_decay = 1e-4  # Decoupled weight decay
        # Счётчик итераций для Adam
        self.iterations = 0
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.W, self.b = [], []
        self.Z_list, self.A_list = [], []
        self.dW_list, self.db_list = [], []
        # Добавим накопители моментов
        self.mW, self.vW, self.mb, self.vb = [], [], [], []
        # Добавим параметры Adam beta1, beta2
        self.beta1, self.beta2 = 0.9, 0.999

        for i in range(len(layer_sizes)-1):
            in_dim, out_dim = layer_sizes[i], layer_sizes[i+1]
            self.W.append(np.random.randn(in_dim, out_dim) * np.sqrt(2/(in_dim + out_dim)))
            self.b.append(np.zeros((1, out_dim)))
            # Инициализация накопителей
            self.mW.append(np.zeros_like(self.W[-1]))
            self.vW.append(np.zeros_like(self.W[-1]))
            self.mb.append(np.zeros_like(self.b[-1]))
            self.vb.append(np.zeros_like(self.b[-1]))

    def _activate(self, Z):
        if self.activation == 'sigmoid':
            return 1/(1+np.exp(-Z))
        elif self.activation == 'relu':
            return np.maximum(0, Z)
        elif self.activation == 'tanh':
            return np.tanh(Z)
        else:
            raise ValueError(self.activation)

    def _dactivate(self, Z):
        if self.activation == 'sigmoid':
            s = 1/(1+np.exp(-Z)); return s*(1-s)
        elif self.activation == 'relu':
            return (Z > 0).astype(float)
        elif self.activation == 'tanh':
            return 1 - np.tanh(Z)**2

    def forward(self, X):
        self.Z_list, self.A_list = [], []
        A = X; self.A_list.append(A)
        for i in range(len(self.W)):
            Z = A.dot(self.W[i]) + self.b[i]; self.Z_list.append(Z)
            A = self._activate(Z) if i < len(self.W)-1 else Z
            self.A_list.append(A)
        return A

    def mse(self, y_pred, y_true):
        return np.mean((y_pred - y_true)**2)

    def backward(self, y_true):
        m = y_true.shape[0]
        # Выходной слой (линейная активация, производная = 1)
        aL, zL = self.A_list[-1], self.Z_list[-1]
        delta = (2*(aL - y_true)/m)
        # Градиенты выхода
        a_prev = self.A_list[-2]
        dW = a_prev.T.dot(delta)
        db = np.sum(delta, axis=0, keepdims=True)
        self.dW_list, self.db_list = [dW], [db]
        # Скрытые слои (используем производную активации)
        for l in range(len(self.layer_sizes)-2, 0, -1):
            z = self.Z_list[l-1]
            a_prev = self.A_list[l-1]
            W_next = self.W[l]
            delta = delta.dot(W_next.T) * self._dactivate(z)
            dW = a_prev.T.dot(delta)
            db = np.sum(delta, axis=0, keepdims=True)
            self.dW_list.insert(0, dW)
            self.db_list.insert(0, db)


    def update_adamw(self, lr):
        self.iterations += 1
        for i in range(len(self.W)):
            self.mW[i] = self.beta1 * self.mW[i] + (1 - self.beta1) * self.dW_list[i]
            self.vW[i] = self.beta2 * self.vW[i] + (1 - self.beta2) * self.dW_list[i] ** 2
            self.mb[i] = self.beta1 * self.mb[i] + (1 - self.beta1) * self.db_list[i]
            self.vb[i] = self.beta2 * self.vb[i] + (1 - self.beta2) * self.db_list[i] ** 2

            mW_hat = self.mW[i] / (1 - self.beta1 ** self.iterations)
            vW_hat = self.vW[i] / (1 - self.beta2 ** self.iterations)
            mb_hat = self.mb[i] / (1 - self.beta1 ** self.iterations)
            vb_hat = self.vb[i] / (1 - self.beta2 ** self.iterations)

            self.W[i] -= self.weight_decay * lr * self.W[i]
            self.W[i] -= lr * mW_hat / (np.sqrt(vW_hat) + self.eps)
            self.b[i] -= lr * mb_hat / (np.sqrt(vb_hat) + self.eps)

# Разбиваем на train/val
perm = np.random.permutation(len(X))
split = int(0.8 * len(X))
X_train, y_train = X[perm[:split]], y[perm[:split]]
X_val,   y_val   = X[perm[split:]], y[perm[split:]]

# Инициализация модели
mlp = MLP([1, 50, 1], activation='tanh')

# Параметры обучения и Reduce on Plateau
initial_lr = 0.01
lr = initial_lr
patience = 20
best_val_loss = np.inf
wait = 0
epochs = 500

train_loss_history = []
val_loss_history   = []

for epoch in range(1, epochs+1):
    # тренировка на всём train
    preds_train = mlp.forward(X_train)
    train_loss = mlp.mse(preds_train, y_train)
    train_loss_history.append(train_loss)

    mlp.backward(y_train)
    mlp.update_adamw(lr)

    # оценка на валидации
    preds_val = mlp.forward(X_val)
    val_loss = mlp.mse(preds_val, y_val)
    val_loss_history.append(val_loss)

    #  Reduce on Plateau
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            lr *= 0.5
            wait = 0
            print(f"Epoch {epoch}: val_loss plateaued, new lr = {lr:.5f}")

    if epoch % 200 == 0:
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, lr={lr:.5f}")

# График финальной аппроксимации
preds_all = mlp.forward(X)
idx = np.argsort(X.flatten())
plt.figure(figsize=(6,4))
plt.scatter(X, y, s=10, alpha=0.5, label='Данные')
plt.plot(X.flatten()[idx], preds_all.flatten()[idx], color='red', linewidth=2, label='MLP AdamW Approximation')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Аппроксимация sin(x)')
plt.legend()
plt.grid(True)
plt.show()
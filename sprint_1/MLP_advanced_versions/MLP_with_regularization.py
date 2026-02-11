import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

class MLP:
    def __init__(self,
                 layer_sizes,
                 activation='relu',
                 lr=0.01,
                 l2=0.0,      # здесь мы будем задавать параметр L2
                 l1=0.0,      # здесь будем задавать параметр L1
                 dropout=0.0  # здесь будем задавать Dropout
                ):
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.W, self.b = [], []
        self.Z_list, self.A_list = [], []
        self.dW_list, self.db_list = [], []
        self.lr = lr
        self.l2 = l2
        self.l1 = l1
        self.dropout = dropout

        for i in range(len(layer_sizes)-1):
            in_dim, out_dim = layer_sizes[i], layer_sizes[i+1]
            self.W.append(np.random.randn(in_dim, out_dim) * np.sqrt(2/(in_dim + out_dim)))
            self.b.append(np.zeros((1, out_dim)))

        # контейнер для масок Dropout (скрытые слои)
        self.masks = [None] * (len(self.W)-1)

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
            s = 1/(1+np.exp(-Z))
            return s*(1-s)
        elif self.activation == 'relu':
            return (Z > 0).astype(float)
        elif self.activation == 'tanh':
            return 1 - np.tanh(Z)**2
        else:
            raise ValueError(self.activation)

    def forward(self, X, train=True):
        self.Z_list, self.A_list = [], []
        A = X; self.A_list.append(A)
        #     self.Z_list.append(Z)
        # выходной слой (линейный)
        for i in range(len(self.W)):
            Z = A.dot(self.W[i]) + self.b[i]; self.Z_list.append(Z)
            A = self._activate(Z) if i < len(self.W)-1 else Z
            if train and self.dropout > 0 and i < len(self.W)-1:
                mask = (np.random.rand(*A.shape) > self.dropout) / (1 - self.dropout)
                A *= mask
                self.masks[i] = mask
            self.A_list.append(A)
        return A

    def mse(self, y_pred, y_true):
        loss = np.mean((y_pred - y_true)**2)
        loss += (self.l2/2) * sum(np.sum(W*W) for W in self.W)
        # Добавьте L1-регуляризацию
        loss += self.l1 * sum(np.sum(np.abs(W)) for W in self.W)
        return loss
    
    def backward(self, y_true):
        m = y_true.shape[0]
        # Выходной слой (линейный)
        aL, zL = self.A_list[-1], self.Z_list[-1]
        delta = (2*(aL - y_true)/m)

        # Градиенты выхода
        a_prev = self.A_list[-2]
        # Добавьте L1-регуляризацию к градиентам
        dW = a_prev.T.dot(delta) + self.l2 * self.W[-1] + self.l1 * np.sign(self.W[-1])
        db = np.sum(delta, axis=0, keepdims=True)
        self.dW_list, self.db_list = [dW], [db]

        # Скрытые слои (используем производную активации)
        for l in range(len(self.layer_sizes)-2, 0, -1):
            z = self.Z_list[l-1]
            a_prev = self.A_list[l-1]
            W_next = self.W[l]
            delta = delta.dot(W_next.T) * self._dactivate(z)
    
            if self.dropout > 0:
                delta *= self.masks[l-1]
            # Добавьте L1-регуляризацию к градиентам        
            dW = a_prev.T.dot(delta) + self.l2 * self.W[l-1] + self.l1 * np.sign(self.W[l-1])
            db = np.sum(delta, axis=0, keepdims=True)
            self.dW_list.insert(0, dW)
            self.db_list.insert(0, db)

    def update_params(self):
        for i in range(len(self.W)):
            self.W[i] -= self.lr * self.dW_list[i]
            self.b[i] -= self.lr * self.db_list[i]

# Функция обучения и сбора кривых loss
def train_model(model, epochs=500):
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        # train
        preds = model.forward(X_train, train=True)
        train_losses.append(model.mse(preds, y_train))
        model.backward(y_train)
        model.update_params()
        # val
        preds_val = model.forward(X_val, train=False)
        val_losses.append(model.mse(preds_val, y_val))
    return train_losses, val_losses

# Пример использования:
m0 = MLP([1,64,64,1], lr=0.01, l2=0.00,    l1=0,    dropout=0.0)  # без регуляризации
m1 = MLP([1,64,64,1], lr=0.01, l2=0.001,   l1=0,    dropout=0.0)  # L2-регуляризация
m2 = MLP([1,64,64,1], lr=0.01, l2=0.00,    l1=0,    dropout=0.1)  # Dropout
m3 = MLP([1,64,64,1], lr=0.01, l2=0.00,    l1=0.001,    dropout=0.0)

l0, v0 = train_model(m0)
l1, v1 = train_model(m1)
l2, v2 = train_model(m2)
l3, v3 = train_model(m3) # L1-регуляризация

# Построение сравнительных кривых
plt.figure(figsize=(8,5))
plt.plot(v0, '--', label='Val (без рег.)')
plt.plot(v1, label='Val (L2.)')
plt.plot(v2, label='Val (Dropout)')
plt.plot(v3, label='Val (L1)')

plt.xlabel('Эпоха')
plt.ylabel('MSE на валидации')
plt.title('Сравнение регуляризаций')
plt.legend()
plt.grid(True)
plt.show()
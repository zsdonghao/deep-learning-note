"""
1. Implement the "todo" parts and make the training converge.
2. Try different loss functions.
"""

import numpy as np

# training data
X = np.array([[0,0,1], [0,1,1], [1,0,1], [1,1,1]]).T
y = np.array([0, 1, 1, 1])

# sigmoid function and its derivative
def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1. - sigmoid(x))

# weight initialization function
def get_weights(shape=()):
    np.random.seed(seed=0)
    return np.random.normal(loc=0.0, scale=0.001, size=shape)

class Network(object):
    def __init__(self, lr=0.01):
        # initialize the network weights
        self.W1 = get_weights((4, 3))
        self.b1 = get_weights((4, 1))
        self.W2 = get_weights((1, 4))
        self.b2 = get_weights((1, 1))
        self.lr = lr

    def forward(self, x):
        # feedforward function
        z1  = np.matmul(self.W1, x) + self.b1 # z^{L-1}
        a1  = sigmoid(z1) # a^{L-1}
        z2  = np.matmul(self.W2, a1) + self.b2 # z^{L}
        a2 = sigmoid(z2) # a^{L}
        return z1, a1, z2, a2
    
    def backward(self, X_batch, y_batch):
        batch_size = 0
        batch_loss = 0
        batch_acc  = 0

        self.grads_W2 = np.zeros_like(self.W2)
        self.grads_b2 = np.zeros_like(self.b2)
        self.grads_W1 = np.zeros_like(self.W1)
        self.grads_b1 = np.zeros_like(self.b1)

        # compute the gradients using a batch of training data
        # for x, y in zip(X_batch, y_batch):
        for i in range(len(X_batch)+1):
            x = X_batch[:, i, np.newaxis]
            y = y_batch[i]

            # feedforward propagation
            z1, a1, z2, a2 = self.forward(x)

            # compute loss
            # 1 : mse
            # loss = 1/2 * np.mean((y-a2)**2) # d_C/ d_a^{L} = (a^{L} - y)
            # 2 : abs
            # loss = np.mean(abs(y - a2))  # |x|' = (1 if x>0 else -1)
            # 3 : sigmoid cross entropy (logistic regression)
            loss = - (y*np.log(a2) + (1-y)*np.log(1-a2)) 
            
            # compute intermediate results (error)
            delta_L = "todo"
            delta_l = "todo"
            
            # compute gradients
            self.grads_W2 += "todo"
            self.grads_b2 += "todo"
            self.grads_W1 += "todo"
            self.grads_b1 += "todo"

            batch_size += 1
            batch_loss += loss
            batch_acc += 1 if (y == (1 if a2>0.5 else 0)) else 0

        self.grads_W2 /= batch_size
        self.grads_b2 /= batch_size
        self.grads_W1 /= batch_size
        self.grads_b1 /= batch_size
        batch_loss /= batch_size
        batch_acc /= batch_size
        print("loss:{} batch_acc:{} batch_size:{} lr:{}".format(batch_loss, batch_acc, batch_size, self.lr))

        # update weights
        self.W2 -= self.lr * self.grads_W2
        self.b2 -= self.lr * self.grads_b2
        self.W1 -= self.lr * self.grads_W1
        self.b1 -= self.lr * self.grads_b1

# instantiate a network
net = Network(lr=10)

# update the network
for i in range(100):
    net.backward(X, y) # input a batch of training data













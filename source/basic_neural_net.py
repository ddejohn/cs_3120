import numpy as np

# Each row is a training example, each column is a feature  [X1, X2, X3]
X = np.array((
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1]),
    dtype=float
)
# y = np.array(([0], [1], [1], [0]), dtype=float)
y=np.array(([0,1],[1,0],[1,0],[0,1]), dtype=float)


# Activation function
def sigmoid(t):
    return 1/(1+np.exp(-t))


# Derivative of sigmoid
def sigmoid_derivative(p):
    return p * (1 - p)


# Class definition
class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        # considering we have 4 nodes in the hidden layer
        self.weights1 = np.random.rand(self.input.shape[1], 6)
        self.weights2 = np.random.rand(6, 2)

        print(f"w1 shape: {self.weights1.shape}")
        print(f"w2 shape: {self.weights2.shape}")
        self.y = y
        self.output = np. zeros(y.shape)

    def __call__(self, X):
        return self.feedforward(X)

    def feedforward(self, X):
        self.layer1 = sigmoid(np.dot(X, self.weights1))
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
        return self.layer2

    def backprop(self):
        diff = (self.y - self.output)
        d_sig_out = sigmoid_derivative(self.output)
        d_weights2 = np.dot(self.layer1.T, 2*diff*d_sig_out)
        d_weights1 = np.dot(
            self.input.T, np.dot(
                2*diff*d_sig_out, self.weights2.T
            )*sigmoid_derivative(self.layer1)
        )

        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def train(self, X, y):
        self.output = self.feedforward(self.input)
        self.backprop()


NN = NeuralNetwork(X, y)
for i in range(10001):  # trains the NN 1,000 times
    if i % 1000 == 0:
        print("for iteration # " + str(i) + "\n")
        print("Input : \n" + str(X))
        print("Actual Output: \n" + str(y))
        print("Predicted Output: \n" + str(NN.feedforward(X)))
        # mean sum squared loss
        print("Loss: \n" + str(np.mean(np.square(y - NN.feedforward(X)))))
        print("\n")

    NN.train(X, y)

X1 = [0., 0., 0.]
X2 = [1., 1., 1.]

y1 = NN(X1)
y2 = NN(X2)

print(y1)
print(y2)

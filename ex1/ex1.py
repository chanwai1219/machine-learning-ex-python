import os
import numpy as np
import matplotlib.pyplot as plt

def featureNormalize(X):
    mu = np.mean(X, 0)
    sigma = np.std(X, 0)
    X = (X - mu) / sigma
    return X, mu, sigma

def computeCost(X, y, theta):
    h = np.dot(X, theta)
    J = np.inner((h - y).T, (h - y).T) / (2 * len(y))
    # J = np.power((h - y), 2).sum() / (2 * len(y))
    return J

def gradientDescent(X, y, theta, alpha, iterations):
    J_history = np.zeros(shape=(iterations, 1))

    for i in range(iterations):
        h = np.dot(X, theta)
        delta = alpha * np.dot(X.T, (h - y)) / len(y)
        theta = theta - delta

        J_history[i, 0] = computeCost(X, y, theta)
    return theta, J_history

def normalEqn(X, y):
    return np.linalg.inv(np.dot(X.T, X)).dot(X.T).dot(y)

path = os.path.dirname(os.path.abspath(__file__)) + '/ex1data1.txt'
data = np.loadtxt(path, delimiter=',')

X = data[:, 0]
y = data[:, 1]

plt.ylabel('Profit in $10,000s')
plt.xlabel('Population of City in 10,000s')
plt.plot(X, y, "rx")
# plt.show()

m = len(X)

theta = np.zeros(shape=(2, 1))
alpha = 0.01
iterations = 1500

X = np.append(np.ones(shape=(X.size, 1)), X.reshape(X.size, 1), 1)
y = y.reshape(y.size, 1)

J = computeCost(X, y, theta)
print(J)

J = computeCost(X, y, np.array([[-1], [2]]))
print(J)

theta, J_history = gradientDescent(X, y, theta, alpha, iterations)
print('Theta computed from gradient descent:', theta, J_history)

# plt.plot(X[:, 1], np.dot(X, theta), '-')
# plt.legend('Training data', 'Linear regression')
# plt.show()

# plt.plot(range(len(J_history)), J_history)
# plt.show()

theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

J_vals = np.zeros(shape=(len(theta0_vals), len(theta1_vals)))

for i, theta0 in enumerate(theta0_vals):
    for j, theta1 in enumerate(theta1_vals):
        t = np.array([[theta0], [theta1]])
        J_vals[i,j] = computeCost(X, y, t)

# plt.contour(theta0_vals, theta1_vals, J_vals, np.logspace(-2, 3, 20))
# plt.xlabel('theta_0')
# plt.ylabel('theta_1')
# plt.scatter(theta[0][0], theta[1][0])
# plt.show()

path = os.path.dirname(os.path.abspath(__file__)) + '/ex1data2.txt'
data = np.loadtxt(path, delimiter=',')
X = data[:, 0:2]
y = data[:, 2]
m = len(y)

print('x =', X[0:10, :], '\ny =', y[0:10])

X, mu, sigma = featureNormalize(X)
print(mu)
print(sigma)

X = np.append(np.ones(shape=(len(X), 1)), X, 1)
y = y.reshape(y.size, 1)

alpha = 0.1
num_iters = 400

theta = np.zeros(shape=(3, 1))
theta, J_history = gradientDescent(X, y, theta, alpha, num_iters)
print(theta)

predict = (np.array([[1650, 3]]) - mu) / sigma
predict = np.append(np.ones(shape=(len(predict), 1)), predict)
price = np.dot(predict, theta)
print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):', price)

# plt.plot(range(len(J_history)), J_history, '-b')
# plt.xlabel('Number of iterations');
# plt.ylabel('Cost J');
# plt.show()

data = np.loadtxt(path, delimiter=',')
X = data[:, 0:2]
y = data[:, 2]

X = np.append(np.ones(shape=(len(X), 1)), X, 1)

theta = normalEqn(X, y)
print(theta)

price = np.dot(np.array([[1, 1650, 3]]), theta.reshape(3,1))
print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):', price)

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op

def sigmoid(z):
    return (1 / (1 + np.exp(-z)))

def costFunction(theta, X, y):
    m, n = X.shape
    theta = theta.reshape((n,1))

    h = sigmoid(np.dot(X, theta))
    J = - (1 / m) * (np.dot(y.T, np.log(h)) + np.dot((1 - y).T, np.log(1-h)))
    grad = (1 / m) * np.dot(X.T, (h - y))

    return J, grad

def costFunctionReg(theta, X, y, l):
    m, n = X.shape
    theta = theta.reshape((n, 1))
    
    reg = (l / m) * theta
    reg[0] = 0

    h = sigmoid(np.dot(X, theta))
    J = -(1 / m) * (np.dot(y.T, np.log(h)) + np.dot((1 - y).T, np.log(1 - h)))
    J = J + (l / (2 * m)) * np.inner(theta[1:].T, theta[1:].T)
    grad = (1 / m) * np.dot(X.T, (h - y)) + reg

    return J, grad

def gradient(theta, X, y):
    m, n = X.shape
    theta = theta.reshape((n,1))

    h = sigmoid(np.dot(X, theta))
    grad = (1 / m) * np.dot(X.T, (h - y))

    return grad.flatten()

def plotDecisionBoundary(theta, X, y):
    if X.shape[1] <= 3:
        pos = (y == 1).flatten()
        neg = (y == 0).flatten()
        plt.plot(X[neg, 1], X[neg, 2], 'ko', label='Not admitted')
        plt.plot(X[pos, 1], X[pos, 2], 'k+', label='Admitted')
        plt.legend(loc='upper right')

        plot_x = np.array([min(X[:,2])-2,  max(X[:,2])+2])
        plot_y = (-1 / theta[2]) * (theta[1] * plot_x + theta[0])
        plt.plot(plot_x, plot_y)

        plt.show()
    else:
        pos = np.where(y == 1)
        neg = np.where(y == 0)

        plt.scatter(X[pos, 1], X[pos, 2], marker='o', c='b')
        plt.scatter(X[neg, 1], X[neg, 2], marker='x', c='r')
        plt.xlabel('Microchip Test 1')
        plt.ylabel('Microchip Test 2')

        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        z = np.zeros(shape=(len(u), len(v)))
        for i in range(len(u)):
            for j in range(len(v)):
                z[i, j] = (mapFeature(np.array(u[i]), np.array(v[j])).dot(np.array(theta)))

        z = z.T
        plt.contour(u, v, z)
        plt.title('lambda = %f' % l)
        plt.legend(['y = 1', 'y = 0', 'Decision boundary'])
        plt.show()
    return

def mapFeature(X1, X2):
    degree = 6

    X1 = X1.reshape(X1.size, 1)
    X2 = X2.reshape(X2.size, 1)

    out = np.ones((X1[:, 0].size, 1))

    for i in range(1, degree + 1):
        for j in range(i + 1):
            # print(i, j, "x1", i-j, "x2", j)
            out = np.hstack((out, (X1 ** (i - j)) * (X2 ** j)))
    return out

path = os.path.dirname(os.path.abspath(__file__)) + '/ex2data1.txt'
data = np.loadtxt(path, delimiter=',')

X = data[:, 0:2]
y = data[:, 2]

m, n = X.shape

# pos = (y == 1)
# neg = (y == 0)

# plt.plot(X[pos, 0], X[pos, 1], 'k+', label='Admitted')
# plt.plot(X[neg, 0], X[neg, 1], 'ko', label='Not admitted')
# plt.legend(loc='upper right')
# plt.show()

X = np.hstack((np.ones(shape=(m, 1)), X))
y = y.reshape(m, 1)

initial_theta = np.zeros(n + 1)

cost, grad = costFunction(initial_theta, X, y)
print('Cost at initial theta (zeros):', cost)
print('Gradient at initial theta (zeros):', grad)

result = op.minimize(fun=costFunction, x0=initial_theta,
                    args=(X, y), method='TNC', jac=True)

print(result)

theta = result.x

# plotDecisionBoundary(theta, X, y)

prob = sigmoid(np.array([1, 45, 85]).dot(theta.T))
print('For a student with scores 45 and 85, we predict an admission probability of ', prob)

h = sigmoid(np.dot(X, theta))
threshold = 0.5
p = np.zeros(y.shape)
p[np.where(h > threshold)] = 1

print('Train Accuracy:', np.mean((p == y)) * 100)


path = os.path.dirname(os.path.abspath(__file__)) + '/ex2data2.txt'
data = np.loadtxt(path, delimiter=',')

X = data[:, 0:2]
y = data[:, 2]

X = mapFeature(X[:, 0], X[:, 1])
y = y.reshape(y.size, 1)
m, n = X.shape

initial_theta = np.zeros(n)
l = 1

cost, grad = costFunctionReg(initial_theta, X, y, l)
print('Cost at initial theta (zeros): ', cost)

result = op.minimize(fun=costFunctionReg, x0=initial_theta,
                    args=(X, y, l), method='TNC', jac=True)

print(result)

theta = result.x
plotDecisionBoundary(theta, X, y)

h = sigmoid(np.dot(X, theta))
threshold = 0.5
p = np.zeros(y.shape)
p[np.where(h > threshold)] = 1

print('Train Accuracy:', np.mean((p == y)) * 100)
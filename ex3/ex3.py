import os
import scipy.io as sio
import numpy as np
import scipy.optimize as op

def sigmoid(z):
    return (1 / (1 + np.exp(-z)))

def lrCostFunction(theta, X, y, l):
    m, n = X.shape
    theta = theta.reshape((n, 1))
    
    reg = (l / m) * theta
    reg[0] = 0

    h = sigmoid(np.dot(X, theta))
    J = -(1 / m) * (np.dot(y.T, np.log(h)) + np.dot((1 - y).T, np.log(1 - h)))
    J = J + (l / (2 * m)) * np.inner(theta[1:].T, theta[1:].T)
    grad = (1 / m) * np.dot(X.T, (h - y)) + reg

    return J, grad

def oneVsAll(X, y, num_labels, l):
    m, n = X.shape

    all_theta = np.zeros((num_labels, n + 1))

    X = np.hstack((np.ones((m, 1)), X))

    for c in range(num_labels):
        initial_theta = np.zeros(n + 1)
        p = np.zeros(y.shape)
        p[np.where(y == c)] = 1
        result = op.minimize(fun=lrCostFunction, x0=initial_theta,
                    args=(X, p, l), method='TNC', jac=True, tol=1e-6)
        all_theta[c, :] = result.x

    return all_theta

def predictOneVsAll(all_theta, X):
    m, n = X.shape

    X = np.hstack((np.ones((m, 1)), X))

    h = sigmoid(np.dot(X, all_theta.T))
    
    return np.argmax(h, 1)

def predict(Theta1, Theta2, X):
    m, n = X.shape

    X = np.hstack((np.ones((m, 1)), X))

    a1 = X
    z2 = np.dot(a1, Theta1.T)
    a2 = sigmoid(z2)
    a2 = np.hstack((np.ones((a2.shape[0], 1)), a2))
    z3 = np.dot(a2, Theta2.T)
    a3 = sigmoid(z3)

    return np.argmax(a3, 1)

path = os.path.dirname(os.path.abspath(__file__)) + '/ex3data1.mat'
data = sio.loadmat(path)

X = data['X']
y = data['y']
y[np.where(y == 10)] = 0

theta_t = np.array([-2, -1, 1, 2])
X_t = np.hstack((np.ones((5, 1)), np.arange(1, 16).reshape(3, 5).T / 10))
y_t = np.array([1, 0, 1, 0, 1]).reshape(5, 1)
lambda_t = 3
J, grad = lrCostFunction(theta_t, X_t, y_t, lambda_t)

print('Cost: %f | Expected cost: 2.534819' % J)
print('Gradients:\n', grad)
print('Expected gradients:\n 0.146561\n -0.548558\n 0.724722\n 1.398003')

num_labels = 10
l = 0.1
all_theta = oneVsAll(X, y, num_labels, l)

pred = predictOneVsAll(all_theta, X)
pred = pred.reshape(pred.size, 1)
print('\nTraining Set Accuracy:', np.mean((pred == y)) * 100)

path = os.path.dirname(os.path.abspath(__file__)) + '/ex3data1.mat'
data = sio.loadmat(path)

X = data['X']
y = data['y']

path = os.path.dirname(os.path.abspath(__file__)) + '/ex3weights.mat'
data = sio.loadmat(path)

Theta1 = data['Theta1']
Theta2 = data['Theta2']

pred = predict(Theta1, Theta2, X)
pred = pred.reshape(pred.size, 1)
# matlab index starts from 1
pred = pred + 1
print('\nTraining Set Accuracy:', np.mean((pred == y)) * 100);
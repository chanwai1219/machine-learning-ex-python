import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op


def linearRegCostFunction(theta, X, y, l):
    m, n = X.shape

    theta = theta.reshape((n, 1))

    tmp = theta[1:]

    j_reg = (l / (2 * m)) * (np.dot(tmp.T, tmp))
    grad_reg = (l / m) * theta
    grad_reg[0] = 0

    h = np.dot(X, theta)

    J = (1 / (2 * m)) * (np.dot((h - y).T, (h - y))) + j_reg

    grad = (1 / m) * np.dot(X.T, (h - y)) + grad_reg

    return J, grad


def trainLinearReg(X, y, l):
    m, n = X.shape

    initial_theta = np.zeros(n)

    result = op.minimize(fun=linearRegCostFunction, x0=initial_theta, args=(
        X, y, l), method='TNC', jac=True)

    # print(result)

    return result.x


def learningCurve(X, y, Xval, yval, l):
    m, n = X.shape

    error_train = np.zeros((m, 1))
    error_val = np.zeros((m, 1))

    for i in range(m):
        X_train = X[:i+1, :]
        y_train = y[:i+1]

        theta = trainLinearReg(X_train, y_train, l)

        error_train[i], _ = linearRegCostFunction(theta, X_train, y_train, 0)
        error_val[i], _ = linearRegCostFunction(theta, Xval, yval, 0)

    return error_train, error_val


def polyFeatures(X, p):
    m, n = X.shape
    X_poly = np.zeros((m, p))

    for i in range(p):
        X_poly[:, i] = X[:, 0] ** (i + 1)

    return X_poly


def featureNormalize(X):
    mu = np.mean(X, 0)
    sigma = np.std(X, 0)
    X = (X - mu) / sigma
    return X, mu, sigma


def plotFit(min_x, max_x, mu, sigma, theta, p):
    x = np.arange((np.min(X) - 15), (np.max(X) + 25), 0.05)
    x = x.reshape(len(x), 1)

    X_poly = polyFeatures(x, p)
    X_poly = X_poly - mu
    X_poly = X_poly / sigma

    X_poly = np.hstack((np.ones((X_poly.shape[0], 1)), X_poly))

    plt.plot(x, np.dot(X_poly, theta), '--', 'LineWidth', 2)

    return


path = os.path.dirname(os.path.abspath(__file__)) + '/ex5data1.mat'
data = sio.loadmat(path)

X = data['X']
y = data['y']
Xtest = data['Xtest']
ytest = data['ytest']
Xval = data['Xval']
yval = data['yval']

m, n = X.shape

# plt.plot(X, y, 'rx')
# plt.xlabel('Change in water level (x)')
# plt.ylabel('Water flowing out of the dam (y)')
# plt.show()

theta = np.array([[1], [1]])
J, grad = linearRegCostFunction(theta, np.hstack((np.ones((m, 1)), X)), y, 1)
print('Cost at theta = [1 ; 1]:', J)

J, grad = linearRegCostFunction(theta, np.hstack((np.ones((m, 1)), X)), y, 1)
print('Gradient at theta = [1 ; 1]:', grad)

l = 0
theta = trainLinearReg(np.hstack((np.ones((m, 1)), X)), y, l)

# plt.plot(X, y, 'rx')
# plt.xlabel('Change in water level (x)')
# plt.ylabel('Water flowing out of the dam (y)')
# plt.plot(X, np.hstack((np.ones((m, 1)), X)).dot(theta) , '--', 'LineWidth', 2)
# plt.show()

l = 0
error_train, error_val = learningCurve(np.hstack((np.ones(
    (X.shape[0], 1)), X)), y, np.hstack((np.ones((Xval.shape[0], 1)), Xval)), yval, l)

# plt.plot(range(m), error_train, label='Train')
# plt.plot(range(m), error_val, label='Cross Validation')
# plt.title('Learning curve for linear regression')
# plt.legend(loc='upper right')
# plt.xlabel('Number of training examples')
# plt.ylabel('Error')
# plt.axis(np.array(([0, 13, 0, 150])))
# plt.show()

print('# Training Examples\tTrain Error\tCross Validation Error')
for i in range(m):
    print('  \t%d\t\t%f\t%f' % (i, error_train[i], error_val[i]))

p = 8

X_poly = polyFeatures(X, p)
X_poly, mu, sigma = featureNormalize(X_poly)
X_poly = np.hstack((np.ones((X_poly.shape[0], 1)), X_poly))

X_poly_test = polyFeatures(Xtest, p)
X_poly_test = X_poly_test - mu
X_poly_test = X_poly_test / sigma
X_poly_test = np.hstack((np.ones((X_poly_test.shape[0], 1)), X_poly_test))

X_poly_val = polyFeatures(Xval, p)
X_poly_val = X_poly_val - mu
X_poly_val = X_poly_val / sigma
X_poly_val = np.hstack((np.ones((X_poly_val.shape[0], 1)), X_poly_val))

print('Normalized Training Example 1:', X_poly[0, :])

l = 0
theta = trainLinearReg(X_poly, y, l)
print(theta)

plt.plot(X, y, 'rx')
plotFit(min(X), max(X), mu, sigma, theta, p)
plt.show()
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.title(print('Polynomial Regression Fit (lambda = %f)' % l))
error_train, error_val = learningCurve(X_poly, y, X_poly_val, yval, l)
plt.plot(range(m), error_train, label='Train')
plt.plot(range(m), error_val, label='Cross Validation')
plt.title(print('Polynomial Regression Learning Curve (lambda = %f)' % l))
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.axis([0, 13, 0, 100])
plt.legend(loc='upper right')
plt.show()

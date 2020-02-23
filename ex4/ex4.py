import os
import scipy.io as sio
import numpy as np
import scipy.optimize as op


def sigmoid(z):
    return (1 / (1 + np.exp(-z)))


def sigmoidGradient(z):
    g = sigmoid(z) * (1 - sigmoid(z))
    return g


def randInitializeWeights(L_in, L_out):
    W = np.zeros((L_out, 1 + L_in))
    epsilon_init = 0.12
    W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init
    return W


def computeNumericalGradient(J, theta):
    numgrad = np.zeros(np.shape(theta))
    perturb = np.zeros(np.shape(theta))
    e = 1e-4
    for p in range(0, np.size(theta)):
        perturb[p] = e
        loss1, para = J(theta - perturb)
        loss2, para = J(theta + perturb)
        # Compute Numerical Gradient
        numgrad[p] = (loss2 - loss1) / (2*e)
        perturb[p] = 0

    return numgrad


def debugInitializeWeights(fan_out, fan_in):
    v = np.arange(1, fan_out*(fan_in+1)+1)
    W = np.sin(v).reshape(fan_out, 1+fan_in) / 10
    return W


def checkNNGradients(*a):
    if len(a) == 0:
        xlambda = 0
    else:
        xlambda = a[0]
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    # We generate some 'random' test data
    Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
    Theta2 = debugInitializeWeights(num_labels, hidden_layer_size)

    # Reusing debugInitializeWeights to generate X
    v = np.arange(1, m+1)
    X = debugInitializeWeights(m, input_layer_size - 1)
    y = 1 + (np.mod(v, num_labels)).T
    m1, n1 = np.shape(Theta1)
    m2, n2 = np.shape(Theta2)

    # Unroll parameters
    nn_params = np.r_[(Theta1.ravel().reshape(m1*n1, 1),
                       Theta2.ravel().reshape(m2*n2, 1))]

    # Short hand for cost function
    def cost_func(p):
        return nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, xlambda)

    cost, grad = cost_func(nn_params)
    numgrad = computeNumericalGradient(cost_func, nn_params)

    # Visually examine the two gradient computations.  The two columns you get should be very similar.
    print(np.c_[numgrad, grad])
    print('The above two columns should be very similar.(Left-Numerical Gradient, Right-Analytical Gradient)')

    return

def nnCostFunction(nn_params,
                   input_layer_size,
                   hidden_layer_size,
                   num_labels,
                   X, y, l):
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, (input_layer_size + 1)))

    Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):],
                        (num_labels, (hidden_layer_size + 1)))

    m, n = X.shape

    J = 0
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    X = np.hstack((np.ones((m, 1)), X))

    tmp = np.zeros((m, num_labels))
    for i in range(m):
        tmp[i, y[i] - 1] = 1

    y = tmp

    tmp1 = Theta1[:, 1:]
    tmp2 = Theta2[:, 1:]
    reg = (l / (2 * m)) * (sum(sum(tmp1**2)) + sum(sum(tmp2**2)))

    a1 = X                                      # a1 -> 5000 x 401
    z2 = np.dot(a1, Theta1.T)                   # z2 -> 5000 x 25
    a2 = sigmoid(z2)
    a2 = np.hstack((np.ones((a2.shape[0], 1)), a2))     # a2 -> 5000 x 26
    z3 = np.dot(a2, Theta2.T)                           # z3 -> 5000 x 10
    h = sigmoid(z3)                                     # h  -> 5000 x 10

    J = -np.log(h).dot(y.T) - np.log(1 - h).dot(1 - y.T)
    J = sum(np.diag(J)) / m
    J = J + reg

    delta3 = h - y
    delta2 = np.dot(delta3, tmp2) * sigmoidGradient(z2)
    Theta2_grad = np.dot(delta3.T, a2)
    Theta1_grad = np.dot(delta2.T, a1)

    Theta1_grad = Theta1_grad / m
    Theta2_grad = Theta2_grad / m

    grad_reg1 = np.hstack((np.zeros((Theta1.shape[0], 1)), (l / m) * tmp1))
    grad_reg2 = np.hstack((np.zeros((Theta2.shape[0], 1)), (l / m) * tmp2))
    Theta1_grad = Theta1_grad + grad_reg1
    Theta2_grad = Theta2_grad + grad_reg2

    grad = np.hstack((Theta1_grad.flatten(), Theta2_grad.flatten()))

    return J, grad

def predict(Theta1,Theta2,X):
    m = X.shape[0]
    num_labels = Theta2.shape[0]

    # You need to return the following variables correctly
    p = np.zeros([m,1])

    # The processing of forward-propagating
    a1 = np.hstack((np.ones([m, 1]), X))  # X is the input layer, so x equals a1
    a2 = sigmoid(np.dot(a1, Theta1.T))  # FP from X to a2
    a2 = np.hstack((np.ones([a2.shape[0], 1]), a2))  # Add a new column for bias
    a3 = sigmoid(np.dot(a2, Theta2.T))  # FP to the output layer

    return np.argmax(a3, 1) + 1

path = os.path.dirname(os.path.abspath(__file__)) + '/ex4data1.mat'
data = sio.loadmat(path)

X = data['X']
y = data['y']

path = os.path.dirname(os.path.abspath(__file__)) + '/ex4weights.mat'
data = sio.loadmat(path)

Theta1 = data['Theta1']
Theta2 = data['Theta2']

input_layer_size = 400
hidden_layer_size = 25
num_labels = 10
nn_params = np.hstack((Theta1.flatten(), Theta2.flatten()))

l = 0

J, grad = nnCostFunction(nn_params, input_layer_size,
                         hidden_layer_size, num_labels, X, y, l)
print('Cost at parameters (loaded from ex4weights, should be 0.287629):', J)

l = 1
J, grad = nnCostFunction(nn_params, input_layer_size,
                         hidden_layer_size, num_labels, X, y, l)
print('Cost at parameters (loaded from ex4weights, should be 0.383770):', J)

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)
initial_nn_params = np.hstack(
    (initial_Theta1.flatten(), initial_Theta2.flatten()))

checkNNGradients()

checkNNGradients(3)

result = op.minimize(fun=nnCostFunction, x0=initial_nn_params, args=(
    input_layer_size, hidden_layer_size, num_labels, X, y, l), method='TNC', jac=True, tol=1e-6)

print(result)

nn_params = result.x
Theta1 = nn_params[:hidden_layer_size * (input_layer_size + 1)].reshape(hidden_layer_size, (input_layer_size + 1))
Theta2 = nn_params[(hidden_layer_size * (input_layer_size + 1)):].reshape(num_labels, (hidden_layer_size + 1))

pred = predict(Theta1, Theta2, X)
acc = np.mean(np.mean((pred==y)*100))
print('Training Set Accuracy: %.4f'%acc)

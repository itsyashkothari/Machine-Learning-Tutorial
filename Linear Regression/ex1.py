import numpy as np
from numpy import loadtxt
import matplotlib.pyplot as plt

'''
%% Machine Learning Online Class - Exercise 1: Linear Regression

%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the
%  linear exercise. You will need to complete the following functions
%  in this exericse:
%
%     warmUpExercise
%     plotData
%     gradientDescent
%     computeCost
%     gradientDescentMulti
%     computeCostMulti
%     featureNormalize
%     normalEqn
%
%  x refers to the population size in 10,000s
%  y refers to the profit in $10,000s
%
%
%% Initialization
'''
print()





''' All functions required for this exercise are defined below '''

def warmUpExercise():
    '''
    %   WARMUPEXERCISE Example function
    %   A = WARMUPEXERCISE() is an example function that returns the 5x5 identity matrix
    %
    %   Instructions: Return the 5x5 identity matrix
    '''
    A = np.identity(5)
    print(A)






def plotData(x, y):
    '''
    %   PLOTDATA Plots the data points x and y into a new figure
    %   PLOTDATA(x,y) plots the data points and gives the figure axes labels of
    %   population and profit.
    %
    %   Instructions: Plot the training data into a figure and set the axes
    %                 labels. Assume the population and revenue data have been
    %                 passed in as the x and y arguments of this function.
    '''
    plt.scatter(x, y, marker='x')
    plt.xlabel("Population of City in 10,000s")
    plt.ylabel("Profit in $10,000s")
    plt.show()





def computeCost(X, y, theta):
    '''
    %  COMPUTECOST Compute cost for linear regression
    %  J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
    %  parameter for linear regression to fit the data points in X and y

    % Initialize some useful values
    '''
    m = y.size  # % number of training examples

    ''' % You need to return the following variables correctly  '''
    J = 0

    '''
    %  Instructions: Compute the cost of a particular choice of theta
    %                You should set J to the cost.
    '''
    # print(X)
    # print(theta)
    A = np.matmul(X, theta)
    A = A.reshape(m)
    # print(A.shape)    //(97,)
    A = np.subtract(A, y)
    # print(y.shape)    //(97,)
    # print()
    # print(A.shape)    // (97,)
    A = np.power(A, 2)
    J = (1*np.sum(A))/(2*m)
    return J




def gradientDescent(X, y, theta, alpha, num_iters):
    '''
    %  GRADIENTDESCENT Performs gradient descent to learn theta
    %  theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by
    %  taking num_iters gradient steps with learning rate alpha

    % Initialize some useful values
    '''
    m = y.size  # % number of training examples
    J_history = np.zeros((num_iters,1))
    # print(J_history.shape)    // (1500,1)
    for iter in range(num_iters):
    # for iter in range(1):
        # print(iter)
        '''
        % ====================== YOUR CODE HERE ======================
        % Instructions: Perform a single gradient step on the parameter vector
        %               theta.
        %
        % Hint: While debugging, it can be useful to print out the values
        %       of the cost function (computeCost) and gradient here.
        '''
        A = np.matmul(X, theta)
        A = A.reshape(97)
        A = np.subtract(A, y)
        A = A.reshape((97,1))
        # print(A)
        # print(A.shape)
        # print(X.shape)
        A = A*X
        # print(A.shape)    // (97,2)
        A = A.sum(axis=0)
        A = A.transpose()
        # print(A.shape)    // (2,)
        # print(theta.shape)    // (2,1)
        theta = theta.reshape(2)
        # print(theta.shape)    // (2,)
        theta = np.subtract(theta, (alpha*A)/m)

        ''' % Save the cost J in every iteration '''
        J_history[iter] = computeCost(X, y, theta)

    return theta






'''
%% ==================== Part 1: Basic Function ====================
% Complete warmUpExercise
'''
print("Running warmUpExercise ... ")
print("5x5 Identity Matrix: ")
print()
warmUpExercise()
print()
print("Program paused. Press any key to continue.")
input()
print()

'''
%% ======================= Part 2: Plotting =======================
'''
print()
print("Plotting Data ...")
data = loadtxt("ex1data1.txt", delimiter=",")

# print(data.shape)    // (97,2)
# print(data)

x = data[:, 0]
y = data[:, 1]

# print(x.shape)    // (97,)
# print(y.shape)    // (97,)

m = y.size  # % number of training examples
# print(m)    // 97

'''
% Plot Data
% Note: You have to complete the code in plotData
'''
plotData(x, y)
print()
print("Program paused. Press any key to continue.")
input()
print()

'''
%% =================== Part 3: Cost and Gradient descent ===================
'''
print()
print()
X = np.column_stack((np.ones(m), x))    # % Add a column of ones to x
# print(X)
theta = np.zeros((2, 1))    # % initialize fitting parameters
# print(theta)


''' % Some gradient descent settings '''
iterations = 1500
alpha = 0.01

print("Testing the cost function ...")
print()


''' % compute and display initial cost '''
# J = 0
J = computeCost(X, y, theta)
print("With theta = [0 ; 0]\nCost computed =", J)
print("Expected cost value (approx) 32.07")
print()

''' % further testing of the cost function '''
newTheta = np.array([-1, 2])
# print(newTheta)
# print(newTheta.shape)    // (2,)
J = computeCost(X, y, newTheta)
print("With theta = [-1 ; 2]\nCost computed =", J)
print("Expected cost value (approx) 54.24")

print()
print("Program paused. Press any key to continue.")
input()



print()
print("Running Gradient Descent ...")
print()
''' % run gradient descent '''
# print(theta)
theta = gradientDescent(X, y, theta, alpha, iterations)

''' % print theta to screen '''
print("Theta found by gradient descent:")
print(theta)
print()
print("Expected theta values (approx)")
print("[-3.6303 1.1664]")

''' % Plot the linear fit '''
plt.scatter(x, y, marker='x')
plt.xlabel("Population of City in 10,000s")
plt.ylabel("Profit in $10,000s")
plt.plot(X[:,1], np.matmul(X, theta), color="red")
plt.legend(['Linear regression', 'Training data'])
plt.show()


''' % Predict values for population sizes of 35,000 and 70,000 '''
print()
print()
predict1 = np.matmul(np.array([1, 3.5]), theta)
print("For population = 35,000, we predict a profit of", predict1*10000)
predict2 = np.matmul(np.array([1, 7]), theta)
print("For population = 70,000, we predict a profit of", predict2*10000)
print()

print("Program paused. Press any key to continue.")
input()
print()



# ''' %% ============= Part 4: Visualizing J(theta_0, theta_1) ============= '''
# print()
# print()
# print("Visualizing J(theta_0, theta_1) ...")
#
# ''' % Grid over which we will calculate J '''

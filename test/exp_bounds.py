'''
Implementation of the ADMM convergence rate SDP from Nishihara et al.,
ICML 2015, equation 11.

Code by Wesley Tansey and Sanmi Koyejo
7/31/2015
'''
import cvxpy as cvx
import numpy as np

kappa = cvx.Parameter(sign="positive")
tau = cvx.Parameter(sign="positive")
alpha = cvx.Parameter(sign="positive")
rho0 = cvx.Parameter(sign="positive")
A_hat = cvx.Parameter(2, 2)
B_hat = cvx.Parameter(2, 2)
C1_hat = cvx.Parameter(2, 2)
C2_hat = cvx.Parameter(2, 2)
D1_hat = cvx.Parameter(2, 2)
D2_hat = cvx.Parameter(2, 2)
M1 = cvx.Parameter(2,2)
M2 = cvx.Parameter(2,2)
c = cvx.Parameter()
zeros = cvx.Parameter(2,2)


c.value = 0 # constant objective function (feasibility only)

# Parameters
kappa.value = 1e4
tau.value = 0.1 # 0 < tau < 1
alpha.value = 1.
rho0.value = 2.

# Set the values of the relevant matrices in the LMI
A_hat.value = np.array([[1, alpha.value - 1], [0, 0]])
B_hat.value = np.array([[alpha.value, -1], [0, -1]])
C1_hat.value = np.array([[-1, -1], [0, 0]])
C2_hat.value = np.array([[1, alpha.value - 1], [0, 0]])
D1_hat.value = np.array([[-1, 0], [1, 0]])
D2_hat.value = np.array([[alpha.value, -1], [0, 1]])
M1.value = np.array([[-2 * rho0.value ** -2, 1.0 / rho0.value * (kappa.value ** -0.5 + kappa.value ** 0.5)],
                     [1. / rho0.value * (kappa.value ** -0.5 + kappa.value ** 0.5), -2]])
M2.value = np.array([[0, 1], [1, 0]])
zeros.value = np.zeros((2,2))

# Variables
P = cvx.Semidef(2)
lam1 = cvx.Variable()
lam2 = cvx.Variable()

# Create the (constant) objective function
obj = cvx.Minimize(c)

# Create the matrices for the LMI
mat1 = cvx.bmat([[A_hat.T * P * A_hat - tau ** 2 * P, A_hat.T * P * B_hat], [B_hat.T * P * A_hat, B_hat.T * P * B_hat]])
mat2 = cvx.bmat([[C1_hat, D1_hat], [C2_hat, D2_hat]])
mat3 = cvx.bmat([[lam1 * M1, zeros], [zeros, lam2 * M2]])

# Create the constraints: LMI, lambda1 and lambda 2 non-negative, P positive definite
constraints = [mat1 + mat2 * mat3 * mat2 << 0, lam1 >= 0, lam2 >= 0, P >> 1e-9]

# Solve the problem
prob = cvx.Problem(obj, constraints)
prob.solve(solver=cvx.CVXOPT)


print('Status: {0}'.format(prob.status))
print('Results: {0}'.format(prob.value))
print('P: {0}'.format(P.value))
print('Lambda1: {0}'.format(lam1.value))
print('Lambda2: {0}'.format(lam2.value))


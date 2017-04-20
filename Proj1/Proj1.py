# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 19:39:20 2017

@author: Ben
"""

import numpy as np
from numpy import linalg as LA
from scipy import optimize
from numpy.linalg import inv

# Importing Data from files
Q = np.loadtxt("Q.txt")
b = np.loadtxt("b.txt")
c = np.loadtxt("c.txt")

# Important Variables
k=0 # Number of iterations
alpha = .9  #Step size
epsilon = 0.001
sigma = 0.001 #10-5 to 10-1
beta = 0.4 #0.1 to 0.5

n=len(b)
x = [0]*n #Starting point of gradient descent

def f(s):  #Defines the quadratic to be evaluated
    return np.matmul(np.matmul(np.transpose(s),Q),s) + np.matmul(np.transpose(b),s) + c


def grad_f(s):  #Defines the gradient of teh given function
    return 2*(np.matmul(Q,s))+b

#Gradient Descent
while True:
    alpha_k=alpha
    while True: #Armijo's Rule
        if f(x-alpha_k*grad_f(x)) <= f(x)-sigma*alpha_k*pow((LA.norm(grad_f(x))),2):
            break
        else:
            alpha_k=beta*alpha_k
    if (LA.norm(grad_f(x))) <= epsilon:
        break
    x = x - alpha_k*grad_f(x)
    k += 1  #Increase the iteration number
    
    
#Displaying the results
print(" ")
print("Part 1: The Results of Gradient Descent:")
print("Epsilon= %f" % epsilon)
print("x*= " + repr(x))
print("f(x*)= %f" % f(x))
print("The Number of Iterations: %i" % k)
print(" ")
print("Part 2: Verifying the Results:")
x_inv = np.matmul(inv(Q),(-b/2))
print("x*= " + repr(x_inv))
print("f(x*)= %f" % f(x_inv))
print(" ")
print("Part 3: Optimization via scipy.optimize:")
m = [0]*n    #Starting point for built in optimize function
xmin = optimize.fmin_bfgs(f,m)
print(xmin)
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 17:17:35 2017

@author: Ben
"""

import math as m
import numpy as np
from numpy import linalg as LA
from scipy import optimize


# Important Variables
alpha = 0.8  #Step size
epsilon = 0.0001
sigma = 0.001 #10-5 to 10-1
beta = 0.4 #0.1 to 0.5

output = [0,0,0,0]
d=[0,0,0]

#==============================================================================
# 
#                           Part 1
# 
# 
#==============================================================================

def grad_f(x, w):#Defines the gradient of the given function
    g = [0, 0, 0]
    g[0] = -(w[0]*(1/x[0]) - 2/((1 - x[0] - x[1] - x[2])**3))
    g[1] = -(w[1]*(1/x[1]) - 2/((1 - x[0] - x[1] - x[2])**3))
    g[2] = -(w[2]*(1/x[2]) - 2/((1 - x[0] - x[1] - x[2])**3))
    return np.array(g)

def f(x, w):
    return -(w[0]*m.log(x[0]) + w[1]*m.log(x[1]) + w[2]*m.log(x[2]) - 1/((1 - x[0] - x[1] - x[2])**2))

def func(w, x_0, alpha, beta, sigma, epsilon):
    k=0 # Number of iterations
    x=x_0
    while True:
        alpha_k=alpha
        x_temp = x - alpha_k*grad_f(x,w)
        while (x_temp[0]<=0)or(x_temp[1]<=0)or(x_temp[2]<=0)or(x_temp[0]+x_temp[1]+x_temp[2]>=1):
            alpha_k=beta*alpha_k
            x_temp = x - alpha_k*grad_f(x,w)
        while True: #Armijo's Rule
            if (f(x-alpha_k*grad_f(x,w), w) <= f(x,w)-sigma*alpha_k*pow((LA.norm(grad_f(x,w))),2)):
                break
            else:
                alpha_k=beta*alpha_k
        if (LA.norm(grad_f(x,w))) <= epsilon:
            break
        x = x - alpha_k*grad_f(x,w)
        k += 1  #Increase the iteration number
    m = [0,0,0,0,0]
    m[0] = f(x,w)
    m[1] = x[0]
    m[2] = x[1]
    m[3] = x[2]
    m[4] = k
    return m

#==============================================================================
# 
#                           Part 1a
# 
# 
#==============================================================================
w = [1, 1, 1]
x_0 = [0.25, 0.25, 0.25]
delta = 0.0000001
output = func(w, x_0, alpha, beta, sigma, epsilon)

d[0] = output[1]
d[1] = output[2] 
d[2] = output[3] 


#Displaying the results
print("***********************************************************************")
print("Part 1a: The Results of Gradient Descent:")
print("Epsilon= %f" % epsilon)
print("Boundary deslta = %f" % delta)
print("w = " + repr(w))
print("x_0 = " + repr(x_0))
print("x* = " + repr(d))
print("f(x*) = %f" % -f(d,w))
print("The Number of Iterations: %i" % output[4])
print(" ")


###### Part d
print(" ")
print("Part 1.d: Optimization via scipy.optimize:")
cons = ({'type': 'ineq', 'fun': lambda x:  -(x[0] + x[1] +x[2] - 1 + delta)},
        {'type': 'ineq', 'fun': lambda x:  (x[0] - delta)},
        {'type': 'ineq', 'fun': lambda x:  (x[1] - delta)},
        {'type': 'ineq', 'fun': lambda x:  (x[2] - delta)})
q = optimize.minimize(f, x_0, args=w, method='SLSQP', constraints=cons)
print(q)
print(" ")
print("f(x*) = %f" % -q.fun)
print("f(x*) = " + repr(-q.x))

#==============================================================================
# 
#                           Part 1b
# 
# 
#==============================================================================
w = [1, 2, 3]
x_0 = [0.25, 0.25, 0.25]
delta = 0.0000001

output = func(w, x_0, alpha, beta, sigma, epsilon)
d[0] = output[1]
d[1] = output[2] 
d[2] = output[3]

#Displaying the results
print("***********************************************************************")
print("Part 1b: The Results of Gradient Descent:")
print("Epsilon= %f" % epsilon)
print("Boundary deslta = %f" % delta)
print("w = " + repr(w))
print("x_0 = " + repr(x_0))
print("x* = " + repr(d))
#print(d)
print("f(x*) = %f" % -f(d,w))
print("The Number of Iterations: %i" % output[4])
print(" ")

###### Part d
print(" ")
print("Part 1.d: Optimization via scipy.optimize:")
cons = ({'type': 'ineq', 'fun': lambda x:  -(x[0] + x[1] +x[2] -1 + delta)},
        {'type': 'ineq', 'fun': lambda x:  (x[0] - delta)},
        {'type': 'ineq', 'fun': lambda x:  (x[1] - delta)},
        {'type': 'ineq', 'fun': lambda x:  (x[2]- delta)})
q = optimize.minimize(f, x_0, args=w, method='SLSQP', constraints=cons)
print(q)
print(" ")
print("f(x*) = %f" % -q.fun)
print("f(x*) = " + repr(-q.x))

#==============================================================================
# 
#                           Part 1c
# 
# 
#==============================================================================
w = [30, 10, 10]
#x_0 = [0.1, 0.6, 0.2]
x_0 = np.random.random(3)
while(x_0[0]+x_0[1]+x_0[2]>=1):
    x_0 = np.random.random(3)

delta = 0.001

output = func(w, x_0, alpha, beta, sigma, epsilon)
d[0] = output[1]
d[1] = output[2] 
d[2] = output[3]

#Displaying the results
print("***********************************************************************")
print("Part 1c: The Results of Gradient Descent:")
print("Epsilon= %f" % epsilon)
print("Boundary deslta = %f" % delta)
print("w = " + repr(w))
print("x_0 = " + repr(x_0))
print("x* = " + repr(d))
#print(d)
print("f(x*) = %f" % -f(d,w))
print("The Number of Iterations: %i" % output[4])
print(" ")

###### Part d
print(" ")
print("Part 1.d: Optimization via scipy.optimize:")
cons = ({'type': 'ineq', 'fun': lambda x:  -(x[0] + x[1] +x[2] -1 + delta)},
        {'type': 'ineq', 'fun': lambda x:  (x[0] - delta)},
        {'type': 'ineq', 'fun': lambda x:  (x[1] - delta)},
        {'type': 'ineq', 'fun': lambda x:  (x[2] - delta)})
q = optimize.minimize(f, x_0, args=w, method='SLSQP', constraints=cons)
print(q)
print(" ")
print("f(x*) = %f" % -q.fun)
print("f(x*) = " + repr(-q.x))

#==============================================================================
# 
#                           Part 2
# 
# 
#==============================================================================

def grad_g(x, w):#Defines the gradient of the given function
    g = [0, 0, 0]
    g[0] = -(w[0]*(1/x[0]) - 2/((1 - x[0] - x[1])**3) - 2/((1 - x[0] - x[2])**3))
    g[1] = -(w[1]*(1/x[1]) - 2/((1 - x[0] - x[1])**3))
    g[2] = -(w[2]*(1/x[2]) - 2/((1 - x[0] - x[2])**3))
    return np.array(g)

def g(x, w):
    return -(w[0]*m.log(x[0]) + w[1]*m.log(x[1]) + w[2]*m.log(x[2]) 
             - 1/((1 - x[0] - x[1])**2)- 1/((1 - x[0] - x[2])**2))

def func2(w, x_0, alpha, beta, sigma, epsilon):
    k=0 # Number of iterations
    x=x_0
    while True:
        alpha_k=alpha
        x_temp = x - alpha_k*grad_g(x,w) 
        while (x_temp[0]<=0)or(x_temp[1]<=0)or(x_temp[2]<=0)or(x_temp[0]+x_temp[1]>=1)or(x_temp[0]+x_temp[2]>=1):
            alpha_k=beta*alpha_k
            x_temp = x - alpha_k*grad_g(x,w)
        while True: #Armijo's Rule
            if (g(x-alpha_k*grad_g(x,w), w) <= g(x,w)-sigma*alpha_k*pow((LA.norm(grad_g(x,w))),2)):
                break
            else:
                alpha_k=beta*alpha_k
        if (LA.norm(grad_g(x,w))) <= epsilon:
            break
        x = x - alpha_k*grad_g(x,w)
        k += 1  #Increase the iteration number
    m = [0,0,0,0,0]
    m[0] = g(x,w)
    m[1] = x[0]
    m[2] = x[1]
    m[3] = x[2]
    m[4] = k
    return m
#==============================================================================
# 
#                           Part 2a
# 
# 
#==============================================================================
w = [1, 1, 1]
x_0 = [0.25, 0.25, 0.25]
delta = 0.0000001

output = func2(w, x_0, alpha, beta, sigma, epsilon)
d[0] = output[1]
d[1] = output[2] 
d[2] = output[3]

#Displaying the results
print("***********************************************************************")
print("Part 2a: The Results of Gradient Descent:")
print("Epsilon= %f" % epsilon)
print("Boundary deslta = %f" % delta)
print("w = " + repr(w))
print("x_0 = " + repr(x_0))
print("x* = " + repr(d))
print("f(x*) = %f" % -g(d,w))
print("The Number of Iterations: %i" % output[4])
print(" ")

###### Part d
print(" ")
print("Part 2.d: Optimization via scipy.optimize:")
cons = ({'type': 'ineq', 'fun': lambda x:  -(x[0] + x[1] - 1 + delta)},
        {'type': 'ineq', 'fun': lambda x:  -(x[0] + x[2] - 1 + delta)},
        {'type': 'ineq', 'fun': lambda x:  (x[0] - delta)},
        {'type': 'ineq', 'fun': lambda x:  (x[1] - delta)},
        {'type': 'ineq', 'fun': lambda x:  (x[2]- delta)})
q = optimize.minimize(g, x_0, args=w, method='SLSQP', constraints=cons)
print(q)
print(" ")
print("f(x*) = %f" % -q.fun)
print("f(x*) = " + repr(-q.x))

#==============================================================================
# 
#                           Part 2b
# 
# 
#==============================================================================
w = [1, 2, 3]
x_0 = [0.25, 0.25, 0.25]
delta = 0.0000001

output = func2(w, x_0, alpha, beta, sigma, epsilon)
d[0] = output[1]
d[1] = output[2] 
d[2] = output[3]

#Displaying the results
print("***********************************************************************")
print("Part 2b: The Results of Gradient Descent:")
print("Epsilon= %f" % epsilon)
print("Boundary deslta = %f" % delta)
print("w = " + repr(w))
print("x_0 = " + repr(x_0))
print("x* = " + repr(d))
print("f(x*) = %f" % -g(d,w))
print("The Number of Iterations: %i" % output[4])
print(" ")

###### Part d
print(" ")
print("Part 2.d: Optimization via scipy.optimize:")
cons = ({'type': 'ineq', 'fun': lambda x:  -(x[0] + x[1] - 1 + delta)},
        {'type': 'ineq', 'fun': lambda x:  -(x[0] + x[2] - 1 + delta)},
        {'type': 'ineq', 'fun': lambda x:  (x[0] - delta)},
        {'type': 'ineq', 'fun': lambda x:  (x[1] - delta)},
        {'type': 'ineq', 'fun': lambda x:  (x[2]- delta)})
q = optimize.minimize(g, x_0, args=w, method='SLSQP', constraints=cons)
print(q)
print(" ")
print("f(x*) = %f" % -q.fun)
print("f(x*) = " + repr(-q.x))

#==============================================================================
# 
#                           Part 2c
# 
# 
#==============================================================================
w = [30, 20, 30]
x_0 = [0.1, 0.6, 0.2]
delta = 0.001

output = func2(w, x_0, alpha, beta, sigma, epsilon)
d[0] = output[1]
d[1] = output[2] 
d[2] = output[3]

#Displaying the results
print("***********************************************************************")
print("Part 2c: The Results of Gradient Descent:")
print("Epsilon= %f" % epsilon)
print("Boundary deslta = %f" % delta)
print("w = " + repr(w))
print("x_0 = " + repr(x_0))
print("x* = " + repr(d))
print("f(x*) = %f" % -g(d,w))
print("The Number of Iterations: %i" % output[4])
print(" ")

###### Part d
print(" ")
print("Part 2.d: Optimization via scipy.optimize:")
cons = ({'type': 'ineq', 'fun': lambda x:  -(x[0] + x[1] - 1 + delta)},
        {'type': 'ineq', 'fun': lambda x:  -(x[0] + x[2] - 1 + delta)},
        {'type': 'ineq', 'fun': lambda x:  (x[0] - delta)},
        {'type': 'ineq', 'fun': lambda x:  (x[1] - delta)},
        {'type': 'ineq', 'fun': lambda x:  (x[2]- delta)})
q = optimize.minimize(g, x_0, args=w, method='SLSQP', constraints=cons)
print(q)
print(" ")
print("f(x*) = %f" % -q.fun)
print("f(x*) = " + repr(-q.x))

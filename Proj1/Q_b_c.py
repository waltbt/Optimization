#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 19:27:21 2017

@author: khaledalshehri
"""

import numpy as np
# Choose n, then generate random values for Q (positive definite),b, and c
n=7
Q=np.random.random((n,n))
Q=Q.dot(Q.T)+0.1*np.identity(n)
b=np.random.random(n)
c=np.random.random(1)
np.savetxt('Q.txt', Q)
np.savetxt('b.txt', b)
np.savetxt('c.txt', c)

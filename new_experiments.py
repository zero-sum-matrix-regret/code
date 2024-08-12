#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import random
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import matplotlib.ticker as plticker
num_eps=8
def plot_with_err(rt, ax, label):
    x = np.arange(1,num_eps+1)
    ci = 1 * np.std(rt, axis=0)
    y = np.mean(rt, axis=0)
    ax.plot(x, y,marker = 'o', label=label)
    ax.fill_between(x, y-ci, y+ci, alpha=.3)
def nash1(A, n):
    A = np.maximum(A, 1e-6)  # Ensure all diagonal elements are at least 1e-6
    den = np.sum(1 / np.diag(A))  # Sum of reciprocals of diagonal elements
    x = 1 / (np.diag(A) * den)  # Compute x directly using NumPy operations
    return x
def adversary(A, x, n):
    # Compute the matrix-vector product in a single step
    column_sums = A @ x  # A multiplied by x (dot product)
    # Find the minimum sum and corresponding index
    min_sum = np.min(column_sums)
    index = np.argmin(column_sums)
    
    return min_sum, index

def generate_diagonal_matrix(n):
    A = np.zeros((n, n))  # Initialize an n*n matrix with zeros
    for i in range(n):
        A[i][i] = 0.4 + 0.2*i / (n-1)  # Set the diagonal elements
    return A

def generate_bernoulli_diagonal_matrix(A):
    n = A.shape[0]
    B = np.zeros((n, n))  # Initialize an n*n matrix with zeros
    
    for i in range(n):
        # Generate Bernoulli random variable for the diagonal element
        B[i][i] = np.random.binomial(1, A[i][i])
    
    return B

def update(A, x1, j, t):
    n = A.shape[0]
    vec = A[:-1, j] - A[-1, j]
    x = x1[:-1]
    adjustment = vec * (1 / t)
    x1[:-1] = np.clip(x1[:-1] + adjustment, 0, 1)
    
    sum1 = np.sum(x1[:-1])
    if sum1 > 1:
        x1[:-1] /= sum1
        x1[-1] = 0
    else:
        x1[-1] = 1 - sum1
        
    return x1
# In[38]:


# UCB 
# Bandit feedback

def ucb(seed):
    print(seed,"ucb")
    rng = np.random.default_rng(seed)
    data1=[]
    n=100
    B=generate_diagonal_matrix(n)
    V=0
    den=0
    for i in range(n):
        den=den+1/B[i][i]
    V=1/den
    for itr in range(num_eps):
        print(itr)
        sum1=0
        flag=0
        T=10**(itr+1)
        B1=np.zeros((n, n))
        for t in range(T):
            if(t%10**7==0):
                print(t)
            x=nash1(B1,n)
            val,index=adversary(B,x,n)
            sum1=sum1+V-val
            Bsamp = generate_bernoulli_diagonal_matrix(B)
            B1= (t/(t+1))*B1+(1/(t+1))*Bsamp
        data1.append(np.log10(sum1))
    return data1


# In[39]:


#our-algo
def ouralgo(seed):
    rng = np.random.default_rng(seed)
    print(seed,"our-algo")
    data2=[]
    n=100
    B=generate_diagonal_matrix(n)
    V=0
    den=0
    for i in range(n):
        den=den+1/B[i][i]
    V=1/den
    for itr in range(num_eps):
        print(itr)
        sum2=0
        flag=0
        T=10**(itr+1)
        B1=np.zeros((n, n))
        B2=np.zeros((n, n))
        jt=0
        x=np.ones(n)/np.sum(np.ones(n))
        count0=1
        t0=1
        error=1
        threshold=min(np.log(T)**2,T**0.5)
        for t in range(T):
            if(t%10**7==0):
                print(t)
            if(count0>0 and t>threshold):
                x=update(B2,x,jt,t0)
                count0=count0-1
            else:
                t0=t+1
                x=nash1(B1,n)
                count0=t0-1
                B2=B1.copy()
            val,jt=adversary(B,x,n)
            sum2=sum2+V-val
            Bsamp = generate_bernoulli_diagonal_matrix(B)
            B1= (t/(t+1))*B1+(1/(t+1))*Bsamp
            sum2=sum2+V-val
            #error=((2*np.log(8*T**2))/(t+1))**0.5
        data2.append(np.log10(sum2))
    return data2


# In[40]:


# EXP3
def hedge(seed):
    print(seed,"hedge")
    rng = np.random.default_rng(seed)
    data3=[]
    n=100
    B=generate_diagonal_matrix(n)
    V=0
    den=0
    for i in range(n):
        den=den+1/B[i][i]
    V=1/den
    weights = np.ones(n)
    for itr in range(num_eps):
        print(itr)
        sum3=0
        flag=0
        T=10**(itr+1)
        eta=(np.log(n)/T)**0.5
        for t in range(T):
            if(t%10**7==0):
                print(t)
            x=weights/np.sum(weights)
            val,index=adversary(B,x,n)
            sum3=sum3+V-val
            Bsamp = generate_bernoulli_diagonal_matrix(B)
            reward_vector = Bsamp[:, index]
            weights *= np.exp(eta * reward_vector)
        data3.append(np.log10(sum3))
    return data3
    
def exp3(seed):
    print(seed, "exp3")
    rng = np.random.default_rng(seed*10)
    data3 = []
    n = 100
    B = generate_diagonal_matrix(n)
    V = 0
    den = 0
    for i in range(n):
        den = den + 1 / B[i][i]
    V = 1 / den
    weights = np.ones(n) 
    gamma = 0.07  # Exploration parameter, you can tune this value
    for itr in range(num_eps):
        print(itr)
        sum3 = 0
        flag = 0
        T = 10 ** (itr + 1)
        eta = (np.log(n) / (T * n)) ** 0.5  # Learning rate
        for t in range(T):
            probabilities = (1 - gamma) * (weights / np.sum(weights)) + gamma / n
            index = rng.choice(n, p=probabilities)
            val, jt = adversary(B, probabilities, n)
            sum3 = sum3 + V - val
            reward = rng.binomial(1, B[index][jt])
            estimated_reward = 1-(1-reward) / probabilities[index]
            weights[index] *= np.exp(eta * estimated_reward)
        data3.append(np.log10(sum3))
    return data3


if __name__ == '__main__':
    print("new-8")
    with Pool() as pool:
        data1 = pool.map(ucb, range(0,5))
        data2 = pool.map(ouralgo, range(0,5))
        data3 = pool.map(hedge, range(0,5))
    print("ucb:",np.mean(data1, axis=0))
    print("our-algo:",np.mean(data2, axis=0))
    print("exp3:",np.mean(data3, axis=0))
    f, ax = plt.subplots()
    plot_with_err(data1, ax, label='UCB')
    plot_with_err(data2, ax, label='Our-Algo')
    plot_with_err(data3, ax, label='Hedge')
    plt.xscale("linear")
    plt.xlabel('Log of Time Step')
    plt.ylabel('Log of Regret')
    plt.legend(loc="upper left")
    loc1=plticker.MultipleLocator(base=0.2)
    loc2=plticker.MultipleLocator(base=1.0)
    ax.xaxis.set_major_locator(loc2)
    ax.yaxis.set_major_locator(loc1)
    ax.grid(which='major', axis='both', linestyle='-')
    plt.show() 
    plt.savefig('fig1.png')       


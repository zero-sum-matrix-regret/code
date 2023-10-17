#!/usr/bin/env python
# coding: utf-8

# In[37]:


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
def nash1(A):
    if(A[0][0]<=A[0][1] and A[0][0]>=A[1][0]):
        return 1,0
    if(A[1][0]<=A[1][1] and A[1][0]>=A[0][0]):
        return 0,1
    if(A[0][1]<=A[0][0] and A[0][1]>=A[1][1]):
        return 1,0
    if(A[1][1]<=A[1][0] and A[1][1]>=A[0][1]):
        return 0,1
    D=A[0][0]-A[0][1]-A[1][0]+A[1][1]
    N1=A[1][1]-A[1][0]
    N2=A[0][0]-A[0][1]
    return N1/D,N2/D

def advnew(A,x1,x2,T):
    if(x1<1/3):
        return 1,0
    if(x1>1/3):
        return 0,1
    return 1/3,2/3

def adv22(A,x1,x2):
    a1=A[0][0]*x1+A[1][0]*x2
    a2=A[0][1]*x1+A[1][1]*x2
    if(a1<=a2):
        return a1
    if(a2<=a1):
        return a2
def adv22gd(A,x1,x2):
    a1=A[0][0]*x1+A[1][0]*x2
    a2=A[0][1]*x1+A[1][1]*x2
    if(a1<=a2):
        return a1,0
    if(a2<=a1):
        return a2,1
    
def val22(A,x0,y0):
    x1=1-x0
    y1=1-y0
    return A[0][0]*x0*y0+A[0][1]*x0*y1+A[1][0]*x1*y0+A[1][1]*x1*y1
def update(A,x1,j,t,error):
	a=A[0][j]-A[1][j]
	D=abs(A[0][0]-A[0][1]-A[1][0]+A[1][1])
	z1,z2=nash1(A)
	if(z1>0 and z2>0):
		xmax=min(1,x1+error/D)
		xmin=max(0,x1-error/D)
		if(a>=0):
			val=min(x1+a*max(1,np.log(t))/(D*t),xmax)
			return val
		else:
			val=max(x1+a*max(1,np.log(t))/(D*t),xmin)
			return val
	elif(a>0):
		val=min(x1+1*max(1,np.log(t))/(2*t),1)
		return val
	elif(a<0):
		val=max(x1-1*max(1,np.log(t))/(2*t),0)
		return val
	else:
		return 1/2
# In[38]:


# UCB 
# Bandit feedback

def ucb(seed):
    print(seed,"ucb")
    rng = np.random.default_rng(seed)
    data1=[]
    for itr in range(num_eps):
        sum1=0
        flag=0
        T=10**(itr+1)
        T1=int(T/2)
        B=[[2/3,0],[0,1/3]]
        V=(B[0][0]*B[1][1]-B[0][1]*B[1][0])/(B[0][0]-B[1][0]-B[0][1]+B[1][1])
        B1=[[0,0],[0,0]]
        U1=[[0,0],[0,0]]
        N1=[[0,0],[0,0]]
        for t in range(T1):
            x1,x2=nash1(U1)
            it=rng.binomial(1,x2)
            y1,y2=advnew(B,x1,x2,T)
            #y1,y2=1/3,2/3
            val=val22(B,x1,y1)
            jt=rng.binomial(1,y2)
            a=rng.binomial(1,B[it][jt])
            B1[it][jt]=(B1[it][jt]*N1[it][jt]+a)/(N1[it][jt]+1)
            sum1=sum1+V-val
            for ui in range(2):
                for uj in range(2):
                    U1[ui][uj]=B1[ui][uj]+((2*np.log(8*T**2))/(N1[ui][uj]+1))**0.5
            N1[it][jt]+=1

        for t in range(T1):
            x1,x2=nash1(U1)
            it=rng.binomial(1,x2)
            val,jt=adv22gd(B,x1,x2)
            a=rng.binomial(1,B[it][jt])
            B1[it][jt]=(B1[it][jt]*N1[it][jt]+a)/(N1[it][jt]+1)
            sum1=sum1+V-val
            for ui in range(2):
                for uj in range(2):
                    U1[ui][uj]=B1[ui][uj]+((2*np.log(8*T**2))/(N1[ui][uj]+1))**0.5
            N1[it][jt]+=1
        data1.append(np.log10(sum1))
    return data1


# In[39]:


#our-algo
def ouralgo(seed):
    rng = np.random.default_rng(seed)
    print(seed,"our-algo")
    data2=[]
    for itr in range(num_eps):
        sum2=0
        flag=0
        T=10**(itr+1)
        T1=int(T/2)
        B=[[2/3,0],[0,1/3]]
        V=(B[0][0]*B[1][1]-B[0][1]*B[1][0])/(B[0][0]-B[1][0]-B[0][1]+B[1][1])
        B2=[[0,0],[0,0]]
        U2=[[0,0],[0,0]]
        F2=[[0,0],[0,0]]
        N2=[[0,0],[0,0]]
        jt=0
        x1=0.5
        count0=1
        t0=1
        error=1
        for t in range(T1):
        	if(count0>0 and t>np.log(T)**2):
        		x1=update(F2,x1,jt,t0,error)
        		count0=count0-1
        	else:
        		t0=t+1
        		F2[0][0]=U2[0][0]
        		F2[0][1]=U2[0][1]
        		F2[1][0]=U2[1][0]
        		F2[1][1]=U2[1][1]
        		z1,z2=nash1(F2)
        		if(z1>0 and z2>0):
        			x1=z1
        		x1=update(F2,x1,jt,t0,error)
        		count0=t0-1
        	x2=1-x1
        	it=rng.binomial(1,x2)
        	y1,y2=advnew(B,x1,x2,T)
        	#y1,y2=1/3,2/3
        	val=val22(B,x1,y1)
        	jt=rng.binomial(1,y2)
        	a=rng.binomial(1,B[it][jt])
        	B2[it][jt]=(B2[it][jt]*N2[it][jt]+a)/(N2[it][jt]+1)
        	sum2=sum2+V-val
        	maxdev=0
        	for ui in range(2):
        		for uj in range(2):
        			dev=((2*np.log(8*T**2))/(N2[ui][uj]+1))**0.5
        			if(maxdev<dev):
        				maxdev=dev
        			U2[ui][uj]=B2[ui][uj]+dev
        	if(maxdev<error):
        		error=maxdev
        	N2[it][jt]+=1
        t1=0
        x1,x2=nash1(B2)
        for t in range(T1):
            x1=update(B2,x1,jt,T1,error)
            x2=1-x1
            if(x1>1 or x1<0):
            	print(x1,t)
            #it=rng.binomial(1,x2)
            val,jt=adv22gd(B,x1,x2)
            #a=rng.binomial(1,B[it][jt])
            #B2[it][jt]=(B2[it][jt]*N2[it][jt]+a)/(N2[it][jt]+1)
            sum2=sum2+V-val
            #for ui in range(2):
            #    for uj in range(2):
            #        U2[ui][uj]=B2[ui][uj]+((2*np.log(8*T**2))/(N2[ui][uj]+1))**0.5
            #N2[it][jt]+=1
        data2.append(np.log10(sum2))
    #print(countx,itr+1,"countx")
    return data2


# In[40]:


# EXP3
def exp3(seed):
    print(seed,"exp3")
    rng = np.random.default_rng(seed)
    data3=[]
    for itr in range(num_eps):
        sum3=0
        flag=0
        T=10**(itr+1)
        T1=int(T/2)
        eta=(np.log(2)/T)**0.5
        W=[0,0]
        B=[[2/3,0],[0,1/3]]
        V=(B[0][0]*B[1][1]-B[0][1]*B[1][0])/(B[0][0]-B[1][0]-B[0][1]+B[1][1])
        B3=[[0,0],[0,0]]
        N3=[[0,0],[0,0]]
        x=[0,0]
        for t in range(T1):
            x[0]=np.exp(-eta*W[0])/(np.exp(-eta*W[0])+np.exp(-eta*W[1]))
            x[1]=1-x[0]
            it=rng.binomial(1,x[1])
            y1,y2=advnew(B,x[0],x[1],T)
            #y1,y2=1/3,2/3
            val=val22(B,x[0],y1)
            jt=rng.binomial(1,y2)
            a=rng.binomial(1,B[it][jt])
            l=(1-a)/(x[it])
            B3[it][jt]=(B3[it][jt]*N3[it][jt]+a)/(N3[it][jt]+1)
            sum3=sum3+V-val
            N3[it][jt]+=1
            W[it]=W[it]+l
            min1=min(W[0],W[1])
            W[0]=W[0]-min1
            W[1]=W[1]-min1

        for t in range(T1):
            x[0]=np.exp(-eta*W[0])/(np.exp(-eta*W[0])+np.exp(-eta*W[1]))
            x[1]=1-x[0]
            it=rng.binomial(1,x[1])
            val,jt=adv22gd(B,x[0],x[1])
            a=rng.binomial(1,B[it][jt])
            l=(1-a)/(x[it])
            B3[it][jt]=(B3[it][jt]*N3[it][jt]+a)/(N3[it][jt]+1)
            sum3=sum3+V-val
            N3[it][jt]+=1
            W[it]=W[it]+l
            min1=min(W[0],W[1])
            W[0]=W[0]-min1
            W[1]=W[1]-min1
            if(jt==0 and flag==0):
                flag=1
        data3.append(np.log10(sum3))
    return data3


# In[42]:

if __name__ == '__main__':
    with Pool() as pool:
      data1 = pool.map(ucb, range(0,128))
      data2 = pool.map(ouralgo, range(0,128))
      data3 = pool.map(exp3, range(0,128))
    print("ucb:",np.mean(data1, axis=0))
    print("our-algo:",np.mean(data2, axis=0))
    print("exp3:",np.mean(data3, axis=0))
    f, ax = plt.subplots()
    plot_with_err(data1, ax, label='UCB')
    plot_with_err(data2, ax, label='Our-Algo')
    plot_with_err(data3, ax, label='EXP3')
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


# In[ ]:





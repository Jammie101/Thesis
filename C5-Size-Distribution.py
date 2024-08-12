import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from numpy import random
import math
import scipy.special
import pandas as pd
import seaborn as sns

# Size of Sub-level L(i,j)
def Size_Of_Sublevel(i,j):
    return (j+1)*(i+1)

# Position of state (s,id,iw,ic) in its sublevel L(s+id,iw+ic). First position is position=0
def Position_State(s,iD,iW,iC):
    return s*(iW+iC+1)+iW

# Sum of rates
def Delta(s,iD,iW,iC):
    return delta*iD+phiD*betaC*s*iC+(betaW*iW+phiW*betaC*iC)*s+betaC*(1-phiW)*iW*iC+betaC*(1-phiW-phiD)*s*iC+(betaC*(1-phiD)*iC+betaW*iW)*iD+rhoW*iW+rhoC*iC


N=12 #N=12 here for running locally, change to N=40 if using HPC
delta=1/4
betaW=3/N #Reproduction number / N (as if in SIR model)
betaC=betaW/3 #Assumed effect on transmission rate
phiW=0.25
phiD=0.25
#epsilonw = probability of recovery of wt-infected individual
#deathw = probability of death of wt-infected individual
deathw = 0.1
epsilonw= 1-deathw
deathc = deathw/3 #Assume some effect on death probability of co-infected individual due to DIP presence
epsilonc= 1-deathc 
rhoW=1
rhoC=1 #Assumed no effect on length of infection from DIP

#Algorithm 5.3 in text

# Define vectors alpha_{i,j}(r,d), for i=0,...,N; j=0,...,N-i; r=0,...,N; d=0,...,N-r. Initialised at all-zeros
alpha=[[[[np.asmatrix(np.zeros((Size_Of_Sublevel(i,j),1))) for d in range(N-r+1)] for r in range(N+1)] for j in range(N-i+1)] for i in range(N+1)]


for r in range(N+1):
    for d in range(N-r+1):
        for iW in range(r+d+1):
            iC=r+d-iW
            if r+d==0:
                alpha[0][r+d][r][d][Position_State(0,0,iW,iC),0]=1
            else:
                if iW>0:
                    alpha[0][r+d][r][d][Position_State(0,0,iW,iC),0]+=betaC*(1-phiW)*iW*iC*alpha[0][r+d][r][d][Position_State(0,0,iW-1,iC+1),0]
                    if d>0:
                        alpha[0][r+d][r][d][Position_State(0,0,iW,iC),0]+=(1-epsilonw)*rhoW*iW*alpha[0][r+d-1][r][d-1][Position_State(0,0,iW-1,iC),0]
                    if r>0:
                        alpha[0][r+d][r][d][Position_State(0,0,iW,iC),0]+=epsilonw*rhoW*iW*alpha[0][r+d-1][r-1][d][Position_State(0,0,iW-1,iC),0]
                if iC>0:
                    if d>0:
                        alpha[0][r+d][r][d][Position_State(0,0,iW,iC),0]+=(1-epsilonc)*rhoC*iC*alpha[0][r+d-1][r][d-1][Position_State(0,0,iW,iC-1),0]
                    if r>0:
                        alpha[0][r+d][r][d][Position_State(0,0,iW,iC),0]+=epsilonc*rhoC*iC*alpha[0][r+d-1][r-1][d][Position_State(0,0,iW,iC-1),0]
                alpha[0][r+d][r][d][Position_State(0,0,iW,iC),0]=alpha[0][r+d][r][d][Position_State(0,0,iW,iC),0]/Delta(0,0,iW,iC)

b=[[[[np.asmatrix(np.zeros((Size_Of_Sublevel(i,j),1))) for d in range(N-r+1)] for r in range(N+1)] for j in range(N-i+1)] for i in range(N+1)]
A=[[np.asmatrix(np.zeros((Size_Of_Sublevel(i,j),Size_Of_Sublevel(i,j)))) for j in range(N-i+1)] for i in range(N+1)]

Sanity_Check=[[np.asmatrix(np.zeros((Size_Of_Sublevel(i,j),1))) for j in range(N-i+1)] for i in range(N+1)]
print("Level 0 Done")
for i in range(1,N+1):
    print(i)
    j=0
    for s in range(i+1):
        iD=i-s
        iW=0
        iC=0
        alpha[i][j][0][0][Position_State(s,iD,iW,iC),0]=1.0
    for j in range(1,N-i+1):
        for s in range(i+1):
            iD=i-s
            for iW in range(j+1):
                iC=j-iW
                if iD>0:
                    A[i][j][Position_State(s,iD,iW,iC),Position_State(s+1,iD-1,iW,iC)]=delta*iD/Delta(s,iD,iW,iC)
                if s>0:
                    A[i][j][Position_State(s,iD,iW,iC),Position_State(s-1,iD+1,iW,iC)]=betaC*phiD*s*iC/Delta(s,iD,iW,iC)
                if iW>0:
                    A[i][j][Position_State(s,iD,iW,iC),Position_State(s,iD,iW-1,iC+1)]=betaC*(1-phiW)*iW*iC/Delta(s,iD,iW,iC)
                for size in range(j,i+j+1):
                    for r in range(size+1):
                        d=size-r
                        if s>0:
                            b[i][j][r][d][Position_State(s,iD,iW,iC),0]+=(betaW*iW+phiW*betaC*iC)*s*alpha[i-1][j+1][r][d][Position_State(s-1,iD,iW+1,iC),0]
                            b[i][j][r][d][Position_State(s,iD,iW,iC),0]+=betaC*(1-phiW-phiD)*s*iC*alpha[i-1][j+1][r][d][Position_State(s-1,iD,iW,iC+1),0]
                        if iD>0:
                            b[i][j][r][d][Position_State(s,iD,iW,iC),0]+=iD*(betaC*(1-phiD)*iC+betaW*iW)*alpha[i-1][j+1][r][d][Position_State(s,iD-1,iW,iC+1),0]
                        if iW>0 and d>0:
                            b[i][j][r][d][Position_State(s,iD,iW,iC),0]+=(1-epsilonw)*rhoW*iW*alpha[i][j-1][r][d-1][Position_State(s,iD,iW-1,iC),0]
                        if iW>0 and r>0:
                            b[i][j][r][d][Position_State(s,iD,iW,iC),0]+=epsilonw*rhoW*iW*alpha[i][j-1][r-1][d][Position_State(s,iD,iW-1,iC),0]
                        if iC>0 and d>0:
                            b[i][j][r][d][Position_State(s,iD,iW,iC),0]+=(1-epsilonc)*rhoC*iC*alpha[i][j-1][r][d-1][Position_State(s,iD,iW,iC-1),0]
                        if iC>0 and r>0:
                            b[i][j][r][d][Position_State(s,iD,iW,iC),0]+=epsilonc*rhoC*iC*alpha[i][j-1][r-1][d][Position_State(s,iD,iW,iC-1),0]
                        b[i][j][r][d][Position_State(s,iD,iW,iC),0]=b[i][j][r][d][Position_State(s,iD,iW,iC),0]/Delta(s,iD,iW,iC)
        for size in range(j,i+j+1):
            for r in range(size+1):
                d=size-r
                alpha[i][j][r][d]=np.linalg.solve(np.eye(Size_Of_Sublevel(i,j))-A[i][j],b[i][j][r][d])
                Sanity_Check[i][j]+=alpha[i][j][r][d]
                #print(alpha[i][j][r][d])
        #print("Sanity check: ",i,j)
        #print(Sanity_Check[i][j])


#%% Heatmaps plot with logged entries 
Test  = [10,1,1,0]
Test2 = [10,1,0,1]

def Matrix_Getter(x):
    A = np.zeros((N+1,N+1))
    for i in range(N+1):
        for j in range(N+1):
            if i + j < x[2]+x[3]:
                continue
            elif i + j > N:
                A[i,j] = np.NAN
            else:
                A[i,j] += alpha[x[0]+x[1]][x[2]+x[3]][i][j][Position_State(x[0],x[1],x[2],x[3])]
    A[0,0] = np.NAN
    return np.log10(A)

IDs = ["25%","50%","75%"]
ISs = [[round(N*0.75-1),round(N*0.25)],[round(N*0.5-1),round(N*0.5)],[round(N*0.25-1),round(N*0.75)]]
print(ISs)
k = int(N/5)
test = Matrix_Getter([2,9,1,0])
print(test)
def Mean_Getter(x):
    a = 0
    b = 0
    for i in range(N+1):
        a += np.nansum(10**x[:,i])*i
        b += np.nansum(10**x[i,:])*i
    return a, b

tm  = Mean_Getter(test)
print(tm)
        
def Plotter(x):
    m = len(x)
    pad=1
    f, axes = plt.subplots(2,m,figsize=(12,6),sharex=True,sharey=True)
    cbar_ax = f.add_axes([.91, .3, .03, .4])
    x1 = [num for num in range(N+1)]
    y1 = [num for num in range(N+1)]
    for i in range(2):
        for j in range(m):
            M = Matrix_Getter([x[j][0],x[j][1],1-i,i])
            df1 = pd.DataFrame(M,columns = y1, index = x1)
            mask1=df1.isnull()
            g = sns.heatmap(df1, mask=mask1, ax = axes[i,j],xticklabels=k,yticklabels=k, cmap="GnBu",cbar=True,square=True,vmin=-4,vmax = 0, cbar_ax=cbar_ax)
            g.invert_yaxis()
            g.tick_params(axis='y', rotation=0)
            L = Mean_Getter(M)
            g.scatter(L[0],L[1],marker = "^", color= 'red',s=25)
            if j ==0:
                g.set_ylabel("$R$ ",fontsize=14,rotation=0)
                if i==0:
                    g.annotate("$i_W(0)=1$", xy=(-0.5, 0.5), xytext=(-pad, 0),
                               xycoords='axes fraction', textcoords='offset fontsize',
                               size='large', ha='center', va='center') 
                if i ==1:
                    g.annotate("$i_C(0)=1$", xy=(-0.5, 0.5), xytext=(-pad, 0),
                               xycoords='axes fraction', textcoords='offset fontsize',
                               size='large', ha='center', va='center')  
            if i ==1:
                g.set_xlabel("$D$",fontsize=14)
            if i==0:
                g.set_title("ID(0)="+str(IDs[j]))
            print(str(i*m+j))
            
    #f.suptitle("N="+str(N),fontsize=18)
    f.tight_layout(rect=[0, 0, .9, 1])
    f.savefig('Size_Dist_Log.png')

Bw = Matrix_Getter(Test)
Bc = Matrix_Getter(Test2)
print(Bw,Bc)
print(Bw[0,1]+Bw[1,0])
print(Bc[0,1]+Bc[1,0])

Plotter(ISs)

#%% Plot Heatmaps with logged entries, conditional on r+d > 1
Test  = [2,9,1,0]
Test2 = [2,9,0,1]

def Cond_Matrix_Getter(x):
    A = np.zeros((N+1,N+1))
    for i in range(N+1):
        for j in range(N+1):
            if i + j < x[2]+x[3]:
                continue
            elif i + j > N:
                A[i,j] = np.NAN
            else:
                A[i,j] += alpha[x[0]+x[1]][x[2]+x[3]][i][j][Position_State(x[0],x[1],x[2],x[3])]
    A[0,0] = np.NAN
    x = A[0,1]
    y = A[1,0]
    A[1,0] = np.NaN
    A[0,1] = np.NaN
    B=A/(1-x-y)
    return np.log10(B)

def Mean_Getter(x):
    a = 0
    b = 0
    for i in range(N+1):
        a += np.nansum(10**x[:,i])*i
        b += np.nansum(10**x[i,:])*i
    return a, b

IDs = ["25%","50%","75%"]
ISs = [[round(N*0.75-1),round(N*0.25)],[round(N*0.5-1),round(N*0.5)],[round(N*0.25-1),round(N*0.75)]]
print(ISs)
k = int(N/5)
def Plotter(x):
    m = len(x)
    pad=2
    f, axes = plt.subplots(2,m,figsize=(12,6),sharex=True,sharey=True)
    cbar_ax = f.add_axes([.91, .3, .03, .4])
    x1 = [num for num in range(N+1)]
    y1 = [num for num in range(N+1)]
    for i in range(2):
        for j in range(m):
            M = Cond_Matrix_Getter([x[j][0],x[j][1],1-i,i])
            df1 = pd.DataFrame(M,columns = y1, index = x1)
            mask1=df1.isnull()
            g = sns.heatmap(df1, mask=mask1, ax = axes[i,j],xticklabels=k,yticklabels=k, cmap="GnBu",cbar=True,square=True,vmin=-4,vmax =0,cbar_ax=cbar_ax)
            g.invert_yaxis()
            g.tick_params(axis='y', rotation=0)
            L = Mean_Getter(M)
            g.scatter(L[0],L[1],marker = "^", color= 'red',s=25)
            if j ==0:
                g.set_ylabel("$R$ ",fontsize=14,rotation=0)
                if i==0:
                    g.annotate("$i_W(0)=1$", xy=(-0.5, 0.5), xytext=(-pad, 0),
                               xycoords='axes fraction', textcoords='offset fontsize',
                               size='large', ha='center', va='center') 
                if i ==1:
                    g.annotate("$i_C(0)=1$", xy=(-0.5, 0.5), xytext=(-pad, 0),
                               xycoords='axes fraction', textcoords='offset fontsize',
                               size='large', ha='center', va='center')  
            if i ==1:
                g.set_xlabel("$D$",fontsize=14)
            if i==0:
                g.set_title("ID(0)="+str(IDs[j]))
            print(str(i*m+j))
    f.tight_layout(rect=[0, 0, .9, 1])
    f.savefig('Conditional_Size_Dist_Log.png')


Bw = Cond_Matrix_Getter(Test)
Bc = Cond_Matrix_Getter(Test2)
print(Bw,Bc)
print(Bw[0,1]+Bw[1,0])
print(Bc[0,1]+Bc[1,0])

Plotter(ISs)
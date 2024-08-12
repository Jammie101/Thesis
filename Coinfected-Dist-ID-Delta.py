MEGAi = 0
#%%
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from numpy import random
import math
import scipy.special

# Size of Sub-level L(i,j)
def Size_Of_Sublevel(i,j):
    return (N-i-j+1)*(i+1)

# Position of state (s,id,iw,ic) in its sublevel L(s+id,iw+ic). First position is position=0
def Position_State(s,iD,iW,iC):
    return s*(N-(s+iD)-iC+1)+iW

# Sum of rates
def Delta(s,iD,iW,iC):
    return delta*iD+phiD*betaC*s*iC+(betaW*iW+phiW*betaC*iC)*s+betaC*(1-phiW)*iW*iC+betaC*(1-phiW-phiD)*s*iC+(betaC*(1-phiD)*iC+betaW*iW)*iD+rhoW*iW+rhoC*iC

N=20
deltas=[1/2,1/4,1/8]
delta = deltas[MEGAi]
betaW=3/N #Reproduction number / N (as if in SIR model)
betaC=betaW/3 #Assume some effect on transmission rate
phiW=0.25
phiD=0.25
#epsilon = probability of recovery
#epsilonw = probability of death
epsilonw = 0.1 #Start with adult population
epsilon= 1-epsilonw
epsilonc = epsilonw/3 #Assume some effect on death probability due to DIP presence
eta= 1-epsilonc
rhoW=1
rhoC=1 #Assumed no effect on length of infection from DIP


# Define vectors xi_{i,j}(n), for n=0,...,N, i=0,...,N; j=0,...,min(N-i,n). Initialised at all-zeros
xi=[[[np.asmatrix(np.zeros((Size_Of_Sublevel(i,j),1))) for j in range(N-i+1)] for i in range(N+1)] for n in range(N+1)]


# Algorithm 1C, to obtain xi's for i=0
for n in range(N+1):
    for iC in reversed(range(n+1)):
        for iW in range(N-iC+1):
            if iC==0:
                if n==0:
                    xi[n][0][iC][Position_State(0,0,iW,iC),0]=1
            else:
                if n>0:
                    if iW>0:
                        xi[n][0][iC][Position_State(0,0,iW,iC),0]+=(betaC*(1-phiW)*iW*iC*xi[n][0][iC+1][Position_State(0,0,iW-1,iC+1),0]+rhoW*iW*xi[n][0][iC][Position_State(0,0,iW-1,iC),0])/Delta(0,0,iW,iC)
                    if iC>0:
                        xi[n][0][iC][Position_State(0,0,iW,iC),0]+=(rhoC*iC*xi[n-1][0][iC-1][Position_State(0,0,iW,iC-1),0])/Delta(0,0,iW,iC)

# We can print the distributions for each possible initial states in L(0)
for j in range(N+1):
    for iW in range(N-j+1):
        print("State ",0,0,iW,j)
        print("Distribution:")
        for n in range(N+1):
            print(n,xi[n][0][j][Position_State(0,0,iW,j),0])
        print("\n")

# We check here that, for any initial state, when you same xi's for different values of n, you get 1 (that is, it is a probability distribution)
for j in range(N+1):
    for iW in range(N-j+1):
        Sanity_Check=0
        print(iW,j)
        for n in range(j,N+1):
            print(n)
            Sanity_Check+=xi[n][0][j][Position_State(0,0,iW,j),0]
        print(Sanity_Check)

# Now, we prepare to implement Algorithm 2. First, we define the matrices and vectors
b=[[[np.asmatrix(np.zeros((Size_Of_Sublevel(i,j),1))) for j in range(N-i+1)] for i in range(N+1)] for n in range(N+1)]
A=[[np.asmatrix(np.zeros((Size_Of_Sublevel(i,j),Size_Of_Sublevel(i,j)))) for j in range(N-i+1)] for i in range(N+1)]

#Sanity_Check=[[np.asmatrix(np.zeros((Size_Of_Sublevel(i,j),1))) for j in range(N-i+1)] for i in range(N+1)]
 
# We build matrices Aij
for i in range(1,N+1):
    for j in range(N-i+1):
        #print(i,j)
        for s in range(i+1):
            iD=i-s
            for iW in range(N-i-j+1):
                iC=j
                #print(s,iD,iW,iC)
                if iD>0 and delta>0:
                    A[i][j][Position_State(s,iD,iW,iC),Position_State(s+1,iD-1,iW,iC)]=delta*iD/Delta(s,iD,iW,iC)
                if s>0 and iC>0:
                    A[i][j][Position_State(s,iD,iW,iC),Position_State(s-1,iD+1,iW,iC)]=betaC*phiD*s*iC/Delta(s,iD,iW,iC)
                if iW>0:
                    #print("hola")
                    #print(Position_State(s,iD,iW,iC),Position_State(s,iD,iW-1,iC))
                    A[i][j][Position_State(s,iD,iW,iC),Position_State(s,iD,iW-1,iC)]=rhoW*iW/Delta(s,iD,iW,iC)
        #print(A[i][j])

for n in range(N+1):
    for i in range(1,N+1):
        for j in reversed(range(min(n,N-i)+1)):
            if i==N:
                if n==0:
                    for s in range(i+1):
                        iD=i-s
                        for iW in range(N-i-j+1):
                            xi[n][i][j][Position_State(s,iD,iW,j),0]=1.0
            else:
                # Here we build vector bij(n)
                for s in range(i+1):
                    iD=i-s
                    for iW in range(N-i-j+1):
                        iC=j
                        if s>0:
                            if iW+iC>0:
                                b[n][i][j][Position_State(s,iD,iW,iC),0]+=(betaW*iW+phiW*betaC*iC)*s*xi[n][i-1][j][Position_State(s-1,iD,iW+1,iC),0]/Delta(s,iD,iW,iC)
                            if iC>0:
                                b[n][i][j][Position_State(s,iD,iW,iC),0]+=betaC*(1-phiW-phiD)*s*iC*xi[n][i-1][j+1][Position_State(s-1,iD,iW,iC+1),0]/Delta(s,iD,iW,iC)
                        if iD>0 and iC+iW>0:
                            b[n][i][j][Position_State(s,iD,iW,iC),0]+=iD*(betaC*(1-phiD)*iC+betaW*iW)*xi[n][i-1][j+1][Position_State(s,iD-1,iW,iC+1),0]/Delta(s,iD,iW,iC)
                        if iW>0 and iC>0:
                            b[n][i][j][Position_State(s,iD,iW,iC),0]+=betaC*(1-phiW)*iW*iC*xi[n][i][j+1][Position_State(s,iD,iW-1,iC+1),0]/Delta(s,iD,iW,iC)
                        if iC>0 and n>0:
                            b[n][i][j][Position_State(s,iD,iW,iC),0]+=rhoC*iC*xi[n-1][i][j-1][Position_State(s,iD,iW,iC-1),0]/Delta(s,iD,iW,iC)
                        if iW+iC+iD==0 and n==0:
                            b[n][i][j][Position_State(s,iD,iW,iC),0]=1.0
                        if iW+iC==0 and n==0 and delta==0:
                            b[n][i][j][Position_State(s,iD,iW,iC),0]=1.0
                xi[n][i][j]=np.linalg.solve(np.eye(Size_Of_Sublevel(i,j))-A[i][j],b[n][i][j])
"""
print("Start")
# We can print the distributions for each possible initial state
for i in range(1,N+1):
    for j in range(N-i+1):
        for s in range(i+1):
            iD=i-s
            for iW in range(N-i-j+1):
                Sanity_Check=0
                print("State ",s,iD,iW,j)
                print("Distribution:")
                for n in range(N+1):
                    print(n,xi[n][i][j][Position_State(s,iD,iW,j),0])
                    print(str(n),str(i),str(j),str(Position_State(s,iD,iW,j)))
                    Sanity_Check+=xi[n][i][j][Position_State(s,iD,iW,j),0]
                print("Suma: ",Sanity_Check)
                print("\n")
"""
def List_Getter(x):
    A  = [0]*(N+1)
    for n in range(N+1):
        A[n] += xi[n][x[0]+x[1]][x[3]][Position_State(x[0],x[1],x[2],x[3])][0,0]
    return A


def Mean_Getter(x):
    M = 0
    for i in range(N+1):
        M += i*x[i]
    return M

xax= np.arange(N+1)
IDs = ["25%","50%","75%"]
ISs = [[round(N*0.75-1),round(N*0.25)],[round(N*0.5-1),round(N*0.5)],[round(N*0.25-1),round(N*0.75)]]

deltalabs = ["2 Weeks","4 Weeks","8 Weeks"]

def Cond_Getter(x):
    K = [0]*(N+1)
    a = 0
    for i in range(1,len(x)):
        a += x[i]
    for j in range(1,len(x)):
        K[j] += x[j]
    return K/a

def list_saver(x):
    k = len(x)
    for j in range(k):
        A1 = List_Getter([x[j][0],x[j][1],1,0])
        A2 = List_Getter([x[j][0],x[j][1],0,1])
        np.savetxt('Dist-IW(0),ID-delta-varied-('+ str(MEGAi + 1) +','+str(j+1) + ').txt',A1)
        np.savetxt('Dist-IC(0),ID-delta-varied-(' +str(MEGAi + 1) +','+str(j+1) + ').txt',A2)

list_saver(ISs)
print(MEGAi)
MEGAi += 1
#%%

def Plotter(x,y):
    m = len(x)
    k = len(y)
    f, axes = plt.subplots(k,m,figsize=(15,15),sharex=True,sharey=True)
    for i in range(k):
        for j in range(m):
            A1 = np.loadtxt('Dist-IW(0),ID-delta-varied-('+ str(i + 1) +','+str(j+1) + ').txt')
            A2 = np.loadtxt('Dist-IC(0),ID-delta-varied-('+ str(i + 1) +','+str(j+1) + ').txt')
            axes[i,j].bar(xax-0.2, A1, width=0.4, label = 'IW(0)=1', color='red')
            axes[i,j].bar(xax+0.2, A2, width=0.4,label = 'IC(0)=1',color='purple')  
            axes[i,j].set_ylim(0,0.65)
            if i ==0:
                axes[i,j].set_title("ID(0) = "+str(x[j]),fontsize=14)
            if j ==0:
                axes[i,j].set_ylabel("$\\frac{1}{\\delta} =$"+str(y[i])+'    '+'Probability',
                                     rotation=0, labelpad=70,fontsize=14)
            if i==0 and j == 2:
                axes[i,j].legend(fontsize=14)
    f.tight_layout(rect=[0, 0, 1, 1])


Plotter(IDs,deltalabs)

#%%
# Finally, one can compute the mean values in order to compare with the algorithms for the mean values
mean_C=[[np.asmatrix(np.zeros((Size_Of_Sublevel(i,j),1))) for j in range(N-i+1)] for i in range(N+1)]

for i in range(N+1):
    for j in range(N-i+1):
        for n in range(N+1):
            mean_C[i][j]+=n*xi[n][i][j]

# We can print the mean values of C for each possible initial state
print("Mean values of C\n")
for i in range(1,N+1):
    for j in range(N-i+1):
        for s in range(i+1):
            iD=i-s
            for iW in range(N-i-j+1):
                Sanity_Check=0
                print("State ",s,iD,iW,j)
                print("Average C: ",mean_C[i][j][Position_State(s,iD,iW,j),0])
                print("\n")

print("End")

"""
# Finally, if Delta=0, one can apply Algorithm 3C
delta=0

xi=[[[np.asmatrix(np.zeros((Size_Of_Sublevel(i,j),1))) for j in range(N-i+1)] for i in range(N+1)] for n in range(N+1)]

# First, we implement Algorithm 1C, to obtain xi's for i=0
for n in range(N+1):
    for iC in reversed(range(n+1)):
        for iW in range(N-iC+1):
            if iC==0:
                if n==0:
                    xi[n][0][iC][Position_State(0,0,iW,iC),0]=1.0
            else:
                if n>0:
                    if iW>0:
                        xi[n][0][iC][Position_State(0,0,iW,iC),0]+=(betaC*(1-phiW)*iW*iC*xi[n][0][iC+1][Position_State(0,0,iW-1,iC+1),0]+rhoW*iW*xi[n][0][iC][Position_State(0,0,iW-1,iC),0])/Delta(0,0,iW,iC)
                    if iC>0:
                        xi[n][0][iC][Position_State(0,0,iW,iC),0]+=(rhoC*iC*xi[n-1][0][iC-1][Position_State(0,0,iW,iC-1),0])/Delta(0,0,iW,iC)

for n in range(N+1):
    for i in range(1,N+1):
        for j in reversed(range(min(n,N-i)+1)):
            if i==N:
                if n==0:
                    for s in range(i+1):
                        iD=i-s
                        for iW in range(N-i-j+1):
                            xi[n][i][j][Position_State(s,iD,iW,j),0]=1.0  
            else:
                for iD in reversed(range(i+1)):
                    s=i-iD
                    for iW in range(N-i-j+1):
                        iC=j
                        if iW+iC==0 and n==0:
                            xi[n][i][j][Position_State(s,iD,iW,iC),0]=1.0
                        else:
                            if iD>0:
                                if iC+iW>0:
                                    xi[n][i][j][Position_State(s,iD,iW,iC),0]+=iD*(betaC*(1-phiD)*iC+betaW*iW)*xi[n][i-1][j+1][Position_State(s,iD-1,iW,iC+1),0]/Delta(s,iD,iW,iC)
                            if s>0:
                                if iC>0:
                                    xi[n][i][j][Position_State(s,iD,iW,iC),0]+=betaC*phiD*s*iC*xi[n][i][j][Position_State(s-1,iD+1,iW,iC),0]/Delta(s,iD,iW,iC)
                                    xi[n][i][j][Position_State(s,iD,iW,iC),0]+=betaC*(1-phiW-phiD)*s*iC*xi[n][i-1][j+1][Position_State(s-1,iD,iW,iC+1),0]/Delta(s,iD,iW,iC)
                                if iW+iC>0:
                                    xi[n][i][j][Position_State(s,iD,iW,iC),0]+=(betaC*phiW*iC+betaW*iW)*s*xi[n][i-1][j][Position_State(s-1,iD,iW+1,iC),0]/Delta(s,iD,iW,iC)
                            if iW>0 and iC>0:
                                xi[n][i][j][Position_State(s,iD,iW,iC),0]+=betaC*(1-phiW)*iC*iW*xi[n][i][j+1][Position_State(s,iD,iW-1,iC+1),0]/Delta(s,iD,iW,iC)
                            if iW>0:
                                xi[n][i][j][Position_State(s,iD,iW,iC),0]+=rhoW*iW*xi[n][i][j][Position_State(s,iD,iW-1,iC),0]/Delta(s,iD,iW,iC)
                            if iC>0 and n>0:
                                xi[n][i][j][Position_State(s,iD,iW,iC),0]+=rhoC*iC*xi[n-1][i][j-1][Position_State(s,iD,iW,iC-1),0]/Delta(s,iD,iW,iC)

#print("Start")
# We can print the distributions for each possible initial state
for i in range(1,N+1):
    for j in range(N-i+1):
        for s in range(i+1):
            iD=i-s
            for iW in range(N-i-j+1):
                Sanity_Check=0
#                print("State ",s,iD,iW,j)
#                print("Distribution:")
                for n in range(N+1):
#                    print(n,xi[n][i][j][Position_State(s,iD,iW,j),0])
                    Sanity_Check+=xi[n][i][j][Position_State(s,iD,iW,j),0]
#                print("Suma: ",Sanity_Check)
#                print("\n")
"""
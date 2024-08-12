import numpy as np
from numba import jit
from scipy.integrate import odeint
#from time import clock

from matplotlib import pyplot as plt
# Load necessary functions
def logSummary(x):

    xNoZero = np.where(x==0, 1, x)

    xLog = np.log(xNoZero)

    xLogMean = np.mean(xLog,axis=0)

    xLogSd = np.sqrt(np.sum((xLog - xLogMean)**2,axis=0)/(len(x)-1))

    return np.exp(xLogMean), np.exp(xLogSd)

 

@jit(nopython=True)

def getTauNumba(x,rates,cR,nuNcr,ratesNcr):

    'computing a candidate time step, tau1'

    if len(cR) == M:

        return np.inf

    else:

        mu = np.absolute(np.dot(ratesNcr,nuNcr))

        sig = np.dot(ratesNcr,nuNcr**2)

        maxx = np.maximum(e*x/g(x),np.ones(N))

        return min(np.min(maxx/mu), np.min(maxx**2/sig))

       

@jit(nopython=True)

def critReactionsNumba(A,x):

    'identifying the critical reactions'

    Y = x/A

    boolY = np.nonzero((Y>-nC) & (Y<0))[0]

    return np.unique(boolY)

   

@jit(nopython=True)

def getTau2Numba(a0C):

    'computing the average time until a critical reaction'

    if a0C:

        return np.random.exponential(1/a0C)

    else:

        return np.inf

 

@jit(nopython=True)

def makeRatesNumba(x):

    'defining the transition rates'

    S, H, E, Be, Nu, TA = x
    A = Sa/(k2*S + Ua)
    
    return np.array([k2*A*S,
                     k3*H,
                     sE,
                     Ue*E + k1*Be*E,
                     k5*Be*(1-Be/BeMAX),
                     k6*E*Be + (k8*Nu*Be)/(1+(TA/ct1)), 
                     (k9*Be*E*N0)/(1+(TA/ct2)) + (k10*Nu*N0)/(1+(TA/ct3)),
                     UN*Nu])
 
@jit(nopython=True)

def chooseNumba(probs):

    'choosing which event occurs'

    u = np.random.rand()

    i = 0

    p_sum = 0.0

    while p_sum < u:

        p_sum += probs[i]

        i += 1

    return i - 1

 

@jit(nopython=True)

def oneStepNumba(x,tt):

    'performing one step of the tau-leaping algorithm'

    rates = makeRatesNumba(x)

    cR = critReactionsNumba(A,x)

    Ncr = np.delete(np.arange(M),cR)

    nuNcr = A[Ncr]

    ratesNcr = rates[Ncr]

    tau1 = getTauNumba(x,rates,cR,nuNcr,ratesNcr)

    if tau1 < 10/np.sum(rates):

        gBool = True

        x, tt, out, tabs = gillespieNumba(x,tt,100)

        return x, tt, out, gBool, tabs

    else:

        gBool = False

        ratesC = rates[cR]

        a0C = np.sum(ratesC)

        tau11 = getTau2Numba(a0C)

        tau = min(tau1,tau11)
        
        condition = 'bad'
        
        while condition == 'bad':
            test = np.zeros(N)
            
            stateChange = np.zeros(N)
        
            for i in range(len(Ncr)):

                stateChange += np.random.poisson(tau*ratesNcr[i])*nuNcr[i]

            if tau==tau11:

                probs = ratesC/a0C
                
                event = chooseNumba(probs)

                stateChange += A[event]
            
            Schange = stateChange[0] 
            stateChange[1] += np.random.binomial(abs(Schange),1/ns)
            stateChange[5] += (k4*(x[3]/(ctb + x[3])) - UTA*x[5])*tau
            for i in range(len(Ncr)):
                test[i] = x[i] + stateChange[i]
            count = 0
            for item in test:
                if item < 0:
                    count += 1
            if count == 0:
                condition = 'good'
            else:
                tau = tau/2
        return x + stateChange, tt + tau, np.zeros((2,1)), gBool, tmax

 

@jit(nopython=True)

def gillespieNumba(x,tt,numruns):

    species = np.zeros((numruns,x.shape[0]))

    ts = np.empty((numruns))

    species[0] = x

    ts[0] = tt

    for j in range(1,numruns):

        rates = makeRatesNumba(x)

        a0 = np.sum(rates)

        if a0 == 0:

            tt = np.inf

            species[j] = x

            ts[j] = tt

        else:

            probs = rates/a0

            index = chooseNumba(probs)

            x = x + A[index]
            if index == 0:
                x[1] += np.random.binomial(1,1/ns)
                
            randomtime = np.random.exponential(1/a0)
            
            tt = tt + randomtime
            # S H E Be N TA
            x[5] = x[5] + (k4*(x[3]/(ctb + x[3])) - UTA*x[5])*randomtime
            #
            species[j] = x

            ts[j] = tt

    tabs = np.max(ts[ts!=np.inf])

    pMin = np.floor(ts[0]/step)+1

    pMax = np.ceil(tabs/step)

    points = step*np.arange(pMin,pMax)

    indices = np.searchsorted(ts,points) - 1

    output = species[indices]
    
    'plotting the actual points from gillespie'

    #plt.plot(ts,species.T[0],color="r")

    #plt.plot(ts,species.T[1],color="r")

    return x, tt, output, tabs

 

@jit(nopython=True)

def oneRunNumba(inits,times):

    species = np.zeros((len(times),inits.shape[0]))

    species[0] = inits

    x, tt = inits, 0

    counter = 1
    
    Infection_Marker = 0

    while counter < len(times):

        while tt < times[counter] and Infection_Marker == 0:
                xcopy = x.copy()

                x, tt, gill, gBool, tabs = oneStepNumba(x,tt)
                
                if x[3] > 10**8: 
                    Infection_Marker += 1
            
        if gBool:

            gillLen = len(gill)

            species[counter:counter+gillLen] = gill[:len(times)-counter]

            counter += gillLen

        else:

            species[counter] = xcopy

            counter += 1

        if np.sum(x) == 0:

            return species, tabs, 0

    return species, 0, Infection_Marker

 

@jit(nopython=True)

def multiRunNumba(inits,times,numruns):

    numSpecies = inits.shape[0]

    absorb = np.empty(numruns)
    
    Dose_Response = np.empty(numruns)

    result = np.empty((numruns*numSpecies,len(times)))

    for i in range(numruns):

        out = oneRunNumba(inits,times)

        result[i*numSpecies:(i+1)*numSpecies] = out[0].T

        absorb[i] = out[1]
        
        Dose_Response[i] = out[2]

    nonzeroInd = np.nonzero(absorb)[0]

    pAbs = len(nonzeroInd)/numruns

    if pAbs:

        mAbs = np.mean(absorb[nonzeroInd])

    else:

        mAbs = np.inf

    return result, pAbs, mAbs, Dose_Response


def makePlot(result,times,numruns,log=True):

    numSpecies = result.shape[0]//numruns

    mean = np.empty((numSpecies,result.shape[1]))

    sd = np.empty((numSpecies,result.shape[1]))

    for i in range(numSpecies):

        speciesInds = np.arange(i,numruns*numSpecies,numSpecies)       

        species = result[speciesInds]

        if log:

            lmean, lsd = logSummary(species)

            mean[i] = lmean

            sd[i] = lsd

            plt.fill_between(times,mean[i]/sd[i],mean[i]*sd[i],color="C%s" % i,alpha=0.2)

        else:

            mean[i] = np.mean(species,axis=0)       

            sd[i] = np.std(species,axis=0,ddof=1)

            plt.fill_between(times,mean[i]-sd[i],mean[i]+sd[i],color="C%s" % i,alpha=0.2)

        plt.plot(times,mean[i],color="C%s" % i)

    plt.ylim(0,)

    plt.xticks(fontsize=14)

    plt.yticks(fontsize=14)

    plt.xlabel("Time",fontsize=18)

    plt.ylabel("Population size",fontsize=18)

    plt.tight_layout()

    return mean, sd

   

#%%

@jit(nopython=True)

def g(x):

    '''Specifying the value of g for each species:

    HOR(i) = highest order of reaction where species i appears as a reactant

    The value of g_i depends on HOR(i) as follows:

        i)   HOR(i) = 1 -> g_i = 1

        ii)  HOR(i) = 2 -> g_i = 2, unless any second order reaction requires

                           two molecules of species i, then g_i = (2 + 1/(x_i-1))

        iii) HOR(i) = 3 -> g_i = 3, unless any third order reaction requires

                           two molecules of species i, then g_i = (3/2)*(2 + 1/(x_i-1))

    '''

    return np.array([1, 1, 2, 2, 2, 1], dtype=np.float64)

 

''' matrix of state change vectors

    - rows of A correspond to the changes in each population for a certain reaction

    i.e. A should have M rows and N columns. '''

''' Transition Matrix'''

 

N = 6       # number of species

M = 8       # number of reactions

nC = 50     # critical threshold

e = 0.03    # tau-leaping epsilon

'''Constants''' 
BeMAX = 5*10**11     #Carrying capacity for the Be Population

ct1 = ct2 = ct3 = 1
ctb = 1000

k1 = 10**-5        #rate of apoptosis
k2= 10**-10 #Host cells phagocyting spores per hour

k3 = 0.05    #Host cells containing spores migrating per hour 

k4 = 2              #Rate of anthrax toxin production by bacteria
k5 = 0.8            #Growth rate of extracellular bacteria
k6 = 5*10**-10     #Rate at which immune cells kill Be
k8 = 6*10**-10     #Rate of phagocytosis and killing of Be by N
k9 = 5*10**-10    #Rate of Neutrophil activation by Be
k10 = 1*10**-5     #Rate of Neutrophil activation by other neutrophils

N0 = 5500           #Source of Neutrophils
nB = 5              #Number of bacteria inside a host cell
ns = 3       #spores per host cell (estimated)
Sa = 3*10**8  #Source of host cells per hour
sE = 1*10**8         #Source of immune cells

Ua = 0.05    #deaths of host cells per hour
Ue = 0.05           #death rate of immune cells
UN = 0.06           #Death rate of Neutrophils
UTA = 2             #Decay of anthrax toxins

numruns = 1000
#Load the initial conditions we are interested in for outcomes and response-probabilties
ICs = [3500, 3750, 4000, 4250, 4500, 4750, 5000,5250]

A = np.array([[-1, 1, 0, 0, 0,0],

              [0, -1, 0, nB, 0,0],
              
              [0, 0, 1, 0, 0,0],
              
              [0, 0, -1, 0, 0,0],
              
              [0, 0, 0, 1, 0,0],
              
              [0, 0, 0, -1, 0,0],
              
              [0, 0, 0, 0, 1,0],
              
              [0, 0, 0, 0, -1,0]
              ],dtype=np.float64)

#Run the tau-leaping algorithm for each of the initial conditions
inits0 = np.array([ICs[0],0,(2*10**9),0,0,0],dtype=np.float64)
inits1 = np.array([ICs[1],0,(2*10**9),0,0,0],dtype=np.float64)
inits2 = np.array([ICs[2],0,(2*10**9),0,0,0],dtype=np.float64)
inits3 = np.array([ICs[3],0,(2*10**9),0,0,0],dtype=np.float64)
inits4 = np.array([ICs[4],0,(2*10**9),0,0,0],dtype=np.float64)
inits5 = np.array([ICs[5],0,(2*10**9),0,0,0],dtype=np.float64)
inits6 = np.array([ICs[6],0,(2*10**9),0,0,0],dtype=np.float64)
inits7 = np.array([ICs[7],0,(2*10**9),0,0,0],dtype=np.float64)
# inits8 = np.array([ICs[8],0,(2*10**9),0,0,0],dtype=np.float64)
# inits9 = np.array([ICs[9],0,(2*10**9),0,0,0],dtype=np.float64)
# inits10 = np.array([ICs[10],0,(2*10**9),0,0,0],dtype=np.float64)


tmax, step = 100, 0.1

times = np.linspace(0,tmax,int(tmax/step)+1)

 

# Timing using the timeit function

#%timeit oneRunNumba(inits,times)



# Running the model once beforehand to get accurate timings when using numba

multiRunNumba(inits0,times,1)


# Running the model and timing how long it takes


#start = clock()

#Run multiple simulations for each of the initial conditions

#for i in range(len(ICs)):
yy0, prob0, mtime0, DR0 = multiRunNumba(inits0,times,numruns)
yy1, prob1, mtime1, DR1 = multiRunNumba(inits1,times,numruns)
yy2, prob2, mtime2, DR2 = multiRunNumba(inits2,times,numruns)
yy3, prob3, mtime3, DR3 = multiRunNumba(inits3,times,numruns)
yy4, prob4, mtime4, DR4 = multiRunNumba(inits4,times,numruns)
yy5, prob5, mtime5, DR5 = multiRunNumba(inits5,times,numruns)
yy6, prob6, mtime6, DR6 = multiRunNumba(inits6,times,numruns)
yy7, prob7, mtime7, DR7 = multiRunNumba(inits7,times,numruns)
# yy8, prob8, mtime8, DR8 = multiRunNumba(inits8,times,numruns)
# yy9, prob9, mtime9, DR9 = multiRunNumba(inits9,times,numruns)
# yy10, prob10, mtime10, DR10 = multiRunNumba(inits10,times,numruns)
#    DRs[i] = sum(DR)

#Dose response list for each initial condition and plot
DR = [np.mean(DR0),np.mean(DR1),np.mean(DR2),np.mean(DR3),np.mean(DR4),np.mean(DR5),np.mean(DR6),np.mean(DR7)]
print(DR)
plt.plot(ICs,DR, marker='o')
# #plt.plot(times,TAa)
plt.plot([4271,4271],[0,1],color = 'k')
plt.ylabel('Probability of Infection (%)')
plt.xlabel('S(0)')
plt.title("Dose Response Curve")
plt.show()

#zz, prob, mtime = multiRunNumba(inits2,times,numruns)

'''yy is the full output of the model stored in an array.

    - The first N rows contain the output for each population for simulation 1

    - The second N rows contain the output for each population for simulation 2 ...'''

#print(clock()-start)

 

''' Using the output from each simulation to find the mean and standard deviation

of each population through time

    - if log is True then the geometric mean and standard deviation will be

    computed, otherwise the usual definition of the mean is used. '''


# #popMean, popSd = makePlot(yy,times,numruns,log=False)

#    Belist.append((yy0[i*6+3]))
    # Sa = np.median([yy6[i*6]],axis=0)
    # Ha= np.median([yy6[i*6+1]],axis=0)
    # Ea= np.median([yy6[i*6+2]],axis=0)
    # Bea= np.median([yy6[i*6+3]],axis=0)
    # Na= np.median([yy6[i*6+4]],axis=0)
    # TAa= np.median([yy6[i*6+5]],axis=0)
    # Belista.append((yy6[i*6+3]))
    
    
#Get medians to compare to simulations
def MedianGetter(yy):
    S = np.zeros((numruns,len(times)))
    H = np.zeros((numruns,len(times)))
    E = np.zeros((numruns,len(times)))
    Be = np.zeros((numruns,len(times)))
    N = np.zeros((numruns,len(times)))
    Ta = np.zeros((numruns,len(times)))
    for i in range(numruns):
        for j in range(len(times)):
            S[i][j] = yy[i*6][j]
            H[i][j] = yy[i*6+1][j]            
            E[i][j] = yy[i*6+2][j]
            Be[i][j] = yy[i*6+3][j]
            N[i][j] = yy[i*6+4][j]
            Ta[i][j] = yy[i*6+5][j]
    return np.median(S,axis=0), np.median(H,axis=0), np.median(E,axis=0), np.median(Be,axis=0), np.median(N,axis=0), np.median(Ta,axis=0), Be

X = MedianGetter(yy1)
        
Infection_Prob = 0
#Infection_Proba = 0
for i in range(numruns):
    if yy3[i*6+3][-1] >= 10**8:
        Infection_Prob += 1
    # if zz[i*6+3][-1] > 10**4:
    #     Infection_Proba += 1

Infection_Prob = Infection_Prob/numruns   
#Infection_Proba = Infection_Proba/numruns   
print(Infection_Prob) 
def Model(z,t):
    S = z[0]
    H = z[1]
    E = z[2]
    Be =z[3]
    N = z[4]
    TA =z[5]
    A = Sa/(k2*S + Ua)

    dSdt = -k2*S*A
    
    dHdt = ((k2*S*A)/ns) - k3*H
    
    dEdt = sE - Ue*E - k1*Be*E
    
    dBedt = k3*nB*H + k5*Be*(1-(Be/BeMAX)) - k6*E*Be - k8*N*Be/(1 + (TA/ct1))
    
    dNdt = k9*Be*E*N0/(1+(TA/ct2)) + k10*N*N0/(1+(TA/ct3)) - UN*N
    
    dTAdt = k4*(Be/(ctb + Be)) - UTA*TA
    
    dzdt = [dSdt,dHdt,dEdt,dBedt,dNdt,dTAdt]
    return dzdt

z0 = [4250,0,(2*10**9),0,0,0]

A = odeint(Model,z0,times)

ylabels = ["Spores, $S(t)$","Host Cells, $H(t)$","Immune Cells, $E(t)$", "Extracellular Bacteria, $B_e(t)$","Neutrophils, $N(t)$","Anthrax Toxins, $T_A(t)$"]
xlabels = ["","","","","Time (Hours)","Time (Hours)"]
#Plot median solution v deterministic
def Plotter(yy,y):
    X = MedianGetter(yy)
    z0 = [ICs[y],0,2*10**9,0,0,0]
    A = odeint(Model,z0,times)
    f1, ax1 = plt.subplots(3,2,figsize=(9,6))
    for i in range(6):
        plt.subplot(3,2,i+1)
        plt.plot(times,X[i])
        plt.plot(times,A[:,i],color='k')
        #plt.plot(times,Sa)
        plt.ylabel(ylabels[i])
        plt.xlabel(xlabels[i])
        if i ==3:
            plt.yscale("log")
        plt.suptitle("Median of 1000 simulations S(0) ="+str(ICs[y]))
        plt.show()
        #plt.savefig('C:\\Users\\Jamie\\Documents\\Judy Day Results\\Tau Leaping\\Median v Deterministic S(0) = '+str(ICs[y]), bbox_inches='tight')
        
yys = [yy0,yy1,yy2,yy3,yy4,yy5,yy6,yy7]

for i in range(len(yys)):
    Plotter(yys[i],i)

X = MedianGetter(yy1)

#     plt.subplot(3,2,2)
#     plt.plot(times,H)
#     plt.plot(times,A[:,1],color='k')
#     plt.ylabel("Host Cells")
    
    
#     plt.subplot(3,2,3)
#     plt.plot(times,E)
#     plt.plot(times,A[:,2],color='k')
#     #plt.plot(times,Ea)
#     plt.ylabel('Immune Cells')
    
#     plt.subplot(3,2,4)
#     plt.plot(times,Be)
#     plt.plot(times,A[:,3],color='k')
#     #plt.plot(times,Bea)
#     plt.ylabel('Extracellular Bacteria')
#     plt.yscale("log")
    
#     plt.subplot(3,2,5)
#     plt.plot(times,N)
#     plt.plot(times,A[:,4],color='k')
#     plt.ylabel('Neutrophils')
#     plt.xlabel('Time')
    
#     plt.subplot(3,2,6)
#     plt.plot(times,TA)
#     plt.plot(times,A[:,5],color='k')
#     #plt.plot(times,TAa)
#     plt.ylabel('Anthrax Toxins')

# plt.xlabel('Time')

# plt.show()
#Plot all simulations for given initial condition
f1, ax1 = plt.subplots(3,2,figsize=(9,6))
for i in range(0,100):
    plt.subplot(3,2,1)
    plt.plot(times,yy4[i*6])
    plt.ylabel(ylabels[0])
    
    plt.subplot(3,2,2)
    plt.plot(times,yy4[i*6+1])
    plt.ylabel(ylabels[1])
    
    
    plt.subplot(3,2,3)
    plt.plot(times,yy4[i*6+2])
    #plt.plot(times,Ea)
    plt.ylabel(ylabels[2])
    
    plt.subplot(3,2,4)
    plt.plot(times,yy4[i*6+3])
    #plt.plot(times,Bea)
    plt.ylabel(ylabels[3])
    plt.yscale("log")
    
    plt.subplot(3,2,5)
    plt.plot(times,yy4[i*6+4])
    plt.ylabel(ylabels[4])
    plt.xlabel('Time')
    
    plt.subplot(3,2,6)
    plt.plot(times,yy4[i*6+5])
    #plt.plot(times,TAa)
    plt.ylabel(ylabels[5])
    
    plt.xlabel('Time (Hours)')
plt.suptitle("1000 Tau-Leaping Realisations for S(0)="+str(ICs[4]))
plt.show()
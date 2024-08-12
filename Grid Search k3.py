import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
time = 2200         #Time for code to run for

k2= 10**-10 #Host cells phagocyting spores per hour
#k3 = 0.05    #Host cells containing spores migrating per hour 

ns = 3              #spores per host cell (estimated)
Sa = 3*10**8        #Source of host cells per hour

Ua = 0.05           #deaths of host cells per hour

t=np.linspace(0,time, time*10)

za = [4500,0]
zb = [9000,0]
zc = [2*10**4,0]
zd = [2*10**7,0]

x = zb

#No Treatment Model
def Model(z,t,k3):
    S = z[0]
    H = z[1]
    A = Sa/(k2*S + Ua)
    dSdt = -k2*S*A
    dHdt = ((k2*S*A)/ns) - k3*H
    
    dzdt = [dSdt,dHdt]
    return dzdt



#Spores (Variable),
#Host Cells, 
#Immune Cells, 
#Extracellular Bacteria,
#Neutrophils
#Anthrax Toxins
times = [960,7200,14400,21600]
def gridsearch(IC):
    tests = [0,0,0,0]
    test = 0
    k3 = 0.0001
    while test == 0:
        z = odeint(Model,IC,t,args=(k3,))
        h = z[:,1]
        for i in range(len(times)):
            if tests[i] == 0:
                if h[times[i]] <= 10:
                     tests[i] = k3
        k3 += 0.0001
        if min(tests) > 0:
            test += 1
    
    return tests

A = gridsearch(x)

for i in range(len(A)):
    z= odeint(Model,x,t,args=(A[i],))
    
    S = z[:,0]
    H = z[:,1]
    # plot results
    
    # plt.subplot(1,2,1)
    # plt.plot(t,S)
    # plt.xlim(-0.1,time)
    # #plt.ylim(0,5100)
    # plt.ylabel('Spores',fontsize=12)
    # plt.xlabel('Time(Hours)',fontsize=12)
    
    # plt.subplot(1,2,2)
    plt.plot(t/24,H,label=str(round(A[i],4)))
    plt.plot([0,100],[10,10],c='k',lw=1)
    for j in range(len(times)):
        plt.plot([times[j]/240,times[j]/240],[0,50],c='k',lw=1)
    #plt.yscale("log")
    plt.xlim(-0.1,time/24)
    #plt.ylim(0,2500)
    plt.ylabel('Host Cells, $H(t)$',fontsize=12)
    plt.xlabel('Time (Days)',fontsize=12)
    plt.legend(title = '$k_3 =$')
    plt.title('$S(0) =$ '+str(x[0]))
# plt.subplot(2,3,3)
# plt.plot(t,E)
# #plt.ylim(0,6200)
# plt.yscale("log")
# plt.xlim(-0.1,time)
# plt.ylabel('Immune Cells',fontsize=12)
# plt.xlabel('Time(Hours)',fontsize=12)

# plt.subplot(2,3,4)
# plt.plot(t,Be)
# plt.xlim(-0.1,time)
# plt.yscale("log")
# plt.ylabel('Extracellular Bacteria',fontsize=12)
# plt.xlabel('Time(Hours)',fontsize=12)

# plt.subplot(2,3,5)
# plt.plot(t,N)
# plt.xlim(-0.1,time)
# plt.yscale("log")
# plt.ylabel('Neutrophils',fontsize=12)
# plt.xlabel('Time(Hours)',fontsize=12)

# plt.subplot(2,3,6)
# plt.plot(t,TA)
# plt.ylim(-0.02,1.02)
# plt.xlim(-0.1,time)
# plt.ylabel('Anthrax Toxins',fontsize=12)   
# plt.xlabel('Time(Hours)',fontsize=12)

plt.show()
#%%
BeMAX = 5*10**11     #Carrying capacity for the Be Population

ct1 = ct2 = ct3 = 1
ctb = 1000

k1 = 10**-5        #rate of apoptosis
k2= 10**-10 #Host cells phagocyting spores per hour

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

def FullModel(z,t):
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


tt = t/24
for i in range(len(A)):
    k3 = A[i]
    y = [x[0],0,(2*10**9),0,0,0]
    z = odeint(FullModel,y,t)

    S = z[:,0]
    H = z[:,1]
    E = z[:,2]
    Be = z[:,3]
    N = z[:,4]
    TA = z[:,5]
    
    
    # plot results
    
    # plt.subplot(2,3,1)
    # plt.plot(t,S)
    # plt.xlim(-0.1,time)
    # #plt.ylim(0,5100)
    # plt.ylabel('Spores',fontsize=12)
    # plt.xlabel('Time(Hours)',fontsize=12)
    
    # plt.subplot(2,3,2)
    # plt.plot(t,H)
    # #plt.yscale("log")
    # plt.xlim(-0.1,time)
    # #plt.ylim(0,2500)
    # plt.ylabel('Host Cells',fontsize=12)
    # plt.xlabel('Time(Hours)',fontsize=12)
    
    # plt.subplot(2,3,3)
    # plt.plot(t,E)
    # #plt.ylim(0,6200)
    # plt.yscale("log")
    # plt.xlim(-0.1,time)
    # plt.ylabel('Immune Cells',fontsize=12)
    # plt.xlabel('Time(Hours)',fontsize=12)
    
    # plt.subplot(2,3,4)
    # plt.plot(t,Be)
    # plt.xlim(-0.1,time)
    # #plt.yscale("log")
    # plt.ylabel('Extracellular Bacteria',fontsize=12)
    # plt.xlabel('Time(Hours)',fontsize=12)
    
    # plt.subplot(2,3,5)
    # plt.plot(t,N)
    # plt.xlim(-0.1,time)
    # #plt.yscale("log")
    # plt.ylabel('Neutrophils',fontsize=12)
    # plt.xlabel('Time(Hours)',fontsize=12)
    
    # plt.subplot(2,3,6)
    # plt.plot(t,TA)
    # plt.ylim(-0.02,1.02)
    # plt.xlim(-0.1,time)
    # plt.ylabel('Anthrax Toxins',fontsize=12)   
    # plt.xlabel('Time(Hours)',fontsize=12)
    
    plt.subplot(2,2,1)
    plt.plot(tt,H)
    #plt.yscale("log")
    plt.xlim(-0.1,time/24)
    #plt.ylim(0,2500)
    plt.ylabel('Host Cells, $H(t)$',fontsize=12)
    plt.xlabel('Time (Days)',fontsize=12)
    
    plt.subplot(2,2,2)
    plt.plot(tt,E)
    #plt.ylim(0,6200)
    plt.yscale("log")
    plt.xlim(-0.1,time/24)
    plt.ylabel('Immune Cells, $E(t)$',fontsize=12)
    plt.xlabel('Time (Days)',fontsize=12)
    
    plt.subplot(2,2,3)
    plt.plot(tt,Be)
    plt.xlim(-0.1,time/24)
    #plt.yscale("log")
    plt.ylabel('Extracellular Bacteria, $B_e(t)$',fontsize=12)
    plt.xlabel('Time (Days)',fontsize=12)
    
    plt.subplot(2,2,4)
    plt.plot(tt,TA)
    plt.ylim(-0.02,1.02)
    plt.xlim(-0.1,time/24)
    plt.ylabel('Anthrax Toxins, $T_A(t)$',fontsize=12)   
    plt.xlabel('Time (Days)',fontsize=12)
    
    plt.suptitle('$S(0) =$ '+str(x[0]))
plt.show()

#%%

x = ['4500', '9000', '$2\\times 10^4$', '$2\\times 10^9$']
y = ['4','30','60','90']

z = np.zeros((4,4))
z[0][0]=1
z[1][0]=1
z[2][0]=1
z[2][1]=1
z[3][0]=1
z[3][1]=1
z[3][2]=1
z[3][3]=1
from matplotlib.lines import Line2D
logos =   [Line2D([0], [0], marker='o', color='w', label='Survival',markerfacecolor='green', markersize=5),
           Line2D([0], [0], marker='o', color='w', label='Death',markerfacecolor='red', markersize=5)]

fig, ax = plt.subplots()
for i in range(len(x)):
    for j in range(len(y)):
        if z[i][j] == 0:
            ax.scatter(x[i], y[j],color = 'green')
        else:
            ax.scatter(x[i], y[j],color = 'red')
            pos = ax.get_position()
ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
ax.legend(handles=logos,loc='center right', bbox_to_anchor=(1.28, 0.5))
plt.ylabel('Approximate PMG (Days)')
plt.xlabel('S(0)')
plt.show()
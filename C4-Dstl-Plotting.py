import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import math
from scipy import integrate
import seaborn as sns
from scipy.integrate import odeint

index = 1
accepted_params_lst = []
accepted_results_lst = []

#load the last iteration of accepted parameter sets for 10 parallel runs of the algorithm
#here the last iteration that all parallel codes returned was 18
for i in range(10):
         accepted_params_lst.append(np.loadtxt('posterior_sorted_iter18_set'+str(i+1)+'.txt'))
         accepted_results_lst.append(np.loadtxt('dists_sorted_iter18_set'+str(i+1)+'.txt'))
accepted_params = np.vstack((accepted_params_lst))
accepted_results = np.vstack((accepted_results_lst))

##### Spore data #####
spore_data = np.loadtxt('C:\\Users\\Jamie\\Documents\\ABC-SMC\\Data\\Dstl Data\\spore_data.txt')
spore_sds = np.loadtxt('C:\\Users\\Jamie\\Documents\\ABC-SMC\\Data\\Dstl Data\\spore_sds.txt')

##### Bacteria data #####
bac_data = np.loadtxt('C:\\Users\\Jamie\\Documents\\ABC-SMC\\Data\\Dstl Data\\bac_data.txt')
bac_sds = np.loadtxt('C:\\Users\\Jamie\\Documents\\ABC-SMC\\Data\\Dstl Data\\bac_sds.txt')

##### PA data #####

PA_data = np.loadtxt('C:\\Users\\Jamie\\Documents\\ABC-SMC\\Data\\Dstl Data\\pa_data.txt')
PA_sds = np.loadtxt('C:\\Users\\Jamie\\Documents\\ABC-SMC\\Data\\Dstl Data\\pa_sds.txt')

##### Define the time points of the data #####
spore_data_times = np.array([0, 1.5, 3.5, 4.25, 5, 6 ,7]) 
spore_data_raw = spore_data_times
bac_data_times = np.array([0, 1.5, 3.5, 4.25, 5, 6 ,7, 16,18,20,22,24,40])
bl_data = 10**bac_data
bac_data_raw = bac_data_times
pa_data_times = np.array([16, 18, 20, 22, 24, 40])
pa_data_raw = pa_data_times
gran = 100 #100 timepoints per hour
t_max = 40
sim_times = np.linspace(0, t_max, t_max*gran +1)
t = sim_times 

#indexes of the data time points in the simulation timecourse, for "matching" with our simulations
spore_data_times = (spore_data_times*gran).astype(int)
bac_data_times = (bac_data_times*gran).astype(int)
pa_data_times = (pa_data_times*gran).astype(int)


#timestep for euler method of ODE integration
dt = 1/gran

#"initial condition" fixed at 4 hours
#"initial condition" fixed at 4 hours
S0 = 10**spore_data[0]
B0 = 10**bac_data[0]
nu0 = 2*10**-4


def diff_eqns(R, t, g, m, lamb, k, alpha):
   return np.array([-g*R[0],
           g*R[0]-m*R[1],
           m*R[1]+lamb*R[2]*(1-(R[2]/k)),
           -alpha*R[2]*R[3]
         ])

def simulation(g, m, lamb, k, alpha, beta, nu, e, tau1, tau2,f):
    g = pow(10,g)
    m = pow(10,m)
    lamb = pow(10,lamb)
    k = pow(10,k)
    alpha = pow(10,alpha)
    beta = pow(10,beta)
    nu = pow(10,nu)
    
    #initial conitions for spore, bacteria and nutrients
    R0 = [f*S0, B0*e, B0*(1-e), 1]
    
    #get the output of the model
    R, infodict = odeint(diff_eqns, R0, sim_times, args=(g, m, lamb, k, alpha),full_output = 1)
    
    #take the outputs of interest
    spore_sim_full = R[:,0]+ (1-f)*S0
    NGB_sim_full = R[:,1]
    veg_sim_full = R[:,2] 
    nut_sim_full = R[:,3]
    
    #manually get the solution for PA
    PA_sim_full = np.zeros(len(sim_times))
    PA_sim_full[0] = 0
    for n in range(0,len(sim_times)-1):
        #if t<tau1 then PA will still be at 0
        if n-tau1*100 < 0:
            PA_sim_full[n+1] = PA_sim_full[n]
        #if tau1<t<tau2, then only production of PA will happen, no protease decay yet
        elif n-tau2*100 < 0:
            PA_sim_full[n+1] = PA_sim_full[n] + dt*(beta*nut_sim_full[int(n-tau1*100)]*veg_sim_full[int(n-tau1*100)] - nu0*PA_sim_full[n])
        #otherwise, t>tau1 and t>tau2 and you have all mechanisms
        elif PA_sim_full[n] == 0:
            PA_sim_full[n+1] = PA_sim_full[n] + dt*(beta*nut_sim_full[int(n-tau1*100)]*veg_sim_full[int(n-tau1*100)])
        else:
            PA_sim_full[n+1] = PA_sim_full[n] + dt*(beta*nut_sim_full[int(n-tau1*100)]*veg_sim_full[int(n-tau1*100)] - nu0*PA_sim_full[n] - np.exp(np.log(nu*veg_sim_full[int(n-tau2*100)]) + np.log(PA_sim_full[n])))
        if PA_sim_full[n+1] < 0:
            PA_sim_full[n+1]=0
    # return logged spores/ml, logged bacterial CFU/ml, PA ng/ml, nutrient proportion and logged newly germinated bacteria (in case it is of interest)
    return np.log10(spore_sim_full), np.log10(NGB_sim_full + veg_sim_full), PA_sim_full, nut_sim_full, np.log10(NGB_sim_full)

#load simulations if previously done or carry them out and save
if os.path.exists('100x20-bac-sims.txt'):
   bac_sims_arr = np.loadtxt('100x20-bac-sims.txt')
   spore_median = np.loadtxt('100x20-spore-median.txt')
   spore_high = np.loadtxt('100x20-spore-high.txt')
   spore_low = np.loadtxt('100x20-spore-low.txt')
   bac_median = np.loadtxt('100x20-bac-median.txt')
   bac_high = np.loadtxt('100x20-bac-high.txt')
   bac_low = np.loadtxt('100x20-bac-low.txt')
   pa_median = np.loadtxt('100x20-pa-median.txt')
   pa_high = np.loadtxt('100x20-pa-high.txt')
   pa_low = np.loadtxt('100x20-pa-low.txt')
   nut_median = np.loadtxt('100x20-nut-median.txt')
   nut_high = np.loadtxt('100x20-nut-high.txt')
   nut_low = np.loadtxt('100x20-nut-low.txt')
   NGB_median = np.loadtxt('100x20-NGB-median.txt')
   NGB_high = np.loadtxt('100x20-NGB-high.txt')
   NGB_low = np.loadtxt('100x20-NGB-low.txt')
else:
    spore_sims = []
    bac_sims = []
    pa_sims = []
    nut_sims = []
    NGB_sims = []
    for i in range(len(accepted_params)):
      print(i)
           
      g, m, lamb, k, alpha, beta, nu, e, tau1, tau2,f  = accepted_params[i,:].T    
      model=simulation(g, m, lamb, k, alpha, beta, nu, e, tau1, tau2,f)
      spore_sims.append(model[0])
      bac_sims.append(model[1])
      pa_sims.append(model[2])
      nut_sims.append(model[3])
      NGB_sims.append(model[4])
    
    spore_sims_arr = np.vstack((spore_sims)).T
    bac_sims_arr = np.vstack((bac_sims)).T
    pa_sims_arr = np.vstack((pa_sims)).T
    nut_sims_arr = np.vstack((nut_sims)).T
    NGB_sims_arr = np.vstack((NGB_sims)).T
    
    np.savetxt('100x20-spore-sims.txt', bac_sims_arr)
    np.savetxt('100x20-bac-sims.txt', bac_sims_arr)
    np.savetxt('100x20-pa-sims.txt', pa_sims_arr)
    np.savetxt('100x20-nut-sims.txt', nut_sims_arr)
    np.savetxt('100x20-NGB-sims.txt', NGB_sims_arr)
    
    spore_median = np.median(spore_sims_arr, axis=1)
    spore_high = np.percentile(spore_sims_arr, 97.5, axis=1)
    spore_low = np.percentile(spore_sims_arr, 2.5, axis=1)
    
    np.savetxt('100x20-spore-median.txt', spore_median)
    np.savetxt('100x20-spore-high.txt', spore_high)
    np.savetxt('100x20-spore-low.txt', spore_low)

    bac_median = np.median(bac_sims_arr, axis=1)
    bac_high = np.percentile(bac_sims_arr, 97.5, axis=1)
    bac_low = np.percentile(bac_sims_arr, 2.5, axis=1)

    np.savetxt('100x20-bac-median.txt', bac_median)
    np.savetxt('100x20-bac-high.txt', bac_high)
    np.savetxt('100x20-bac-low.txt', bac_low)

    pa_median = np.median(pa_sims_arr, axis=1)
    pa_high = np.percentile(pa_sims_arr, 97.5, axis=1)
    pa_low = np.percentile(pa_sims_arr, 2.5, axis=1)

    np.savetxt('100x20-pa-median.txt', pa_median)
    np.savetxt('100x20-pa-high.txt', pa_high)
    np.savetxt('100x20-pa-low.txt', pa_low)
    
    nut_median = np.median(nut_sims_arr, axis=1)
    nut_high = np.percentile(nut_sims_arr, 97.5, axis=1)
    nut_low = np.percentile(nut_sims_arr, 2.5, axis=1)

    np.savetxt('100x20-nut-median.txt', nut_median)
    np.savetxt('100x20-nut-high.txt', nut_high)
    np.savetxt('100x20-nut-low.txt', nut_low)
    
    NGB_median = np.median(NGB_sims_arr, axis=1)
    NGB_high = np.percentile(NGB_sims_arr, 97.5, axis=1)
    NGB_low = np.percentile(NGB_sims_arr, 2.5, axis=1)

    np.savetxt('100x20-NGB-median.txt', NGB_median)
    np.savetxt('100x20-NGB-high.txt', NGB_high)
    np.savetxt('100x20-NGB-low.txt', NGB_low)

#%% Posteriors figure
labels = ('$g$','$m$', '$\lambda$', '$K$', '$\\alpha$','$\\beta$', '$\\nu$', '$\\epsilon$', '$\\tau_1$', '$\\tau_2$', '$f$')
labelspost = ('$log_{10} g$ $(h)^{-1}$','$log_{10}m$ $(h^{-1})$', '$log_{10}\lambda$ $(h^{-1})$', '$log_{10}K$ $(CFU)$', '$log_{10}\\alpha$ $(CFU\\cdot h)^{-1}$','$log_{10}\\beta$ $ng (CFU \\cdot h)^{-1}$', '$log_{10}\\nu$ $(CFU \\cdot h)^{-1}$', '$\\epsilon$', '$\\tau_1$ $(h)$', '$\\tau_2$ $(h)$', '$f$')

#        g, m, lamb, K, alph, beta,nu,e, tau1, tau2, f
mins = [-3, -3, -1, 6, -12, -7, -15, 0,  0,  0, 0]
maxs = [1, 1,   1, 9,  -3,  0,  0,  1, 15, 24, 1]
numb = 10000
locs = [1,1,1,1,1,1,1,1,1,1,1]

f1, ax1 = plt.subplots(2,6,figsize=(12,4),dpi=120)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.35, hspace=0.3)

plt.subplot(2,6,1)
prior = np.random.uniform(mins[7],maxs[7],numb)
sns.kdeplot(prior, shade=True, color='gray', alpha=0.6, label='Prior')
sns.kdeplot(accepted_params[:,7], shade=True, color='green', alpha=0.6, label=labelspost[7])
plt.grid()
plt.xlabel(str(labelspost[7]))
plt.ylabel('Density', fontsize=10)

for i in range(4):
   plt.subplot(2,6,i+2)
   prior = np.random.uniform(mins[i+1],maxs[i+1],numb)
   sns.kdeplot(prior, shade=True, color='gray', alpha=0.6, label='Prior')
   sns.kdeplot(accepted_params[:,i+1], shade=True, color='green', alpha=0.6, label=labelspost[i])
   plt.grid()
   if i == 4:
       plt.ylabel('Density', fontsize=10)
   plt.xlabel(str(labelspost[i+1]))

plt.subplot(2,6,6)
prior = np.random.uniform(mins[-1],maxs[-1],numb)
sns.kdeplot(prior, shade=True, color='gray', alpha=0.6, label='Prior')
sns.kdeplot(accepted_params[:,-1], shade=True, color='green', alpha=0.6, label=labelspost[-1])
plt.grid()
plt.xlabel(str(labelspost[-1]))
plt.xlim(0.98,1.001)
plt.ylabel('Density', fontsize=10)

plt.subplot(2,6,7)
prior = np.random.uniform(mins[5],maxs[5],numb)
sns.kdeplot(prior, shade=True, color='gray', alpha=0.6, label='Prior')
sns.kdeplot(accepted_params[:,5], shade=True, color='green', alpha=0.6, label=labelspost[5])
plt.grid()
plt.xlabel(str(labelspost[5]))
plt.ylabel('Density', fontsize=10)

plt.subplot(2,6,8)
plt.axis('off')

plt.subplot(2,6,9)
prior = np.random.uniform(mins[6],maxs[6],numb)
sns.kdeplot(prior, shade=True, color='gray', alpha=0.6, label='Prior')
sns.kdeplot(accepted_params[:,6], shade=True, color='green', alpha=0.6, label=labelspost[5])
plt.grid()
plt.xlabel(str(labelspost[6]))
plt.ylabel('Density', fontsize=10)

for i in range(2):
   plt.subplot(2,6,i+10)
   prior = np.random.uniform(mins[i+8],maxs[i+8],numb)
   sns.kdeplot(prior, shade=True, color='gray', alpha=0.6, label='Prior')
   sns.kdeplot(accepted_params[:,i+8], shade=True, color='green', alpha=0.6, label=labelspost[i+6])
   plt.grid()
   plt.xlabel(str(labelspost[i+8]))
   plt.ylabel('Density', fontsize=10)

plt.subplot(2,6,12)
prior = np.random.uniform(mins[0],maxs[0],numb)
sns.kdeplot(prior, shade=True, color='gray', alpha=0.6, label='Prior')
sns.kdeplot(accepted_params[:,0], shade=True, color='green', alpha=0.6, label=labelspost[0])
plt.grid()
plt.xlabel(str(labelspost[0]))
plt.ylabel('Density', fontsize=10)
plt.show()

lis = []
for i in range(len(accepted_results)):
               if accepted_params[i][9] < 10:
                   lis.append(i)                   
print(len(lis))
#%% Plot predictions for spores, bacteria and PA
fig = plt.subplots(1,3,figsize=(10,3.5),dpi=120)
plt.subplots_adjust(wspace=0.3)

plt.subplot(1,3,1)
plt.plot(t, spore_median, label='Median',color='red')
plt.fill_between(t, spore_high, spore_low, alpha=0.3, label='95% CI',color='red')
plt.scatter(spore_data_raw,  spore_data, label='Mean data', color='red',zorder=2)
plt.errorbar(spore_data_raw, spore_data, yerr=np.squeeze(spore_sds), capsize=0, ls='none', color='red')
#plt.legend(loc='upper right', fontsize=12)
#plt.yscale('log')
plt.ylabel('Spore count (Log$_{10}$ Spores/ml)', fontsize=12)
plt.xlabel('Time (hours)', fontsize=12)
plt.xlim(-1, 40.5)
#plt.ylim(0.5,5)

plt.subplot(1,3,2)
plt.plot(t, bac_median, label='Median',color='blue')
plt.fill_between(t, bac_high, bac_low, alpha=0.3, label='95% CI',color='blue')
plt.scatter(bac_data_raw,  bac_data, label='Mean data', color='blue',zorder=2)
plt.errorbar(bac_data_raw, bac_data, yerr=np.squeeze(bac_sds), capsize=0, ls='none', color='blue')
#plt.legend(loc='upper right', fontsize=12)
#plt.yscale('log')
plt.ylabel('Bacterial count (Log$_{10}$ CFU/ml)', fontsize=12)
plt.xlabel('Time (hours)', fontsize=12)
plt.xlim(-1, 40.5)
#plt.ylim(0.5,5)


plt.subplot(1,3,3)
plt.plot(t, pa_median, label='Median',color='orange')
plt.fill_between(t, pa_high, pa_low, alpha=0.3, label='95% CI',color='orange')
plt.scatter(pa_data_raw, PA_data, label='Mean data', color='orange',zorder=2)
plt.errorbar(pa_data_raw, PA_data, yerr=np.squeeze(PA_sds), capsize=0, ls='none', color='orange')
#plt.legend(loc='lower right', fontsize=12)
#plt.yscale('log')
plt.ylabel('PA concentration ng/ml)', fontsize=12)
plt.xlabel('Time (hours)', fontsize=12)
plt.xlim(-1,40.5)
"""
fig = plt.subplots(1,2,figsize=(7,3),dpi=120)
plt.subplots_adjust(wspace=0.3)

plt.subplot(1,2,1)
plt.plot(t, NGB_median, label='Median',color='purple')
plt.fill_between(t, NGB_high, NGB_low, alpha=0.3, label='95% CI',color='purple')
#plt.legend(loc='upper right', fontsize=12)
#plt.yscale('log')
plt.ylabel('NGB count (CFU/ml)', fontsize=12)
plt.xlabel('Time (hours)', fontsize=12)
#plt.ylim(10E0,)
plt.xlim(0, 40.5)
#plt.ylim(0.5,5)

plt.subplot(1,2,2)
plt.plot(t, nut_median, label='Median', color='green')
plt.fill_between(t, nut_high, nut_low, alpha=0.3, label='95% CI',color='green')
#plt.legend(loc='lower right', fontsize=12)
#plt.yscale('log')
plt.ylabel('Nutrient Proportion', fontsize=12)
plt.xlabel('Time (hours)', fontsize=12)
plt.xlim(0,40.5)
"""
#%% Plot correlation coefficients

dfpam = pd.DataFrame(accepted_params, columns=labels) #Create pandas dataframe
dfpam2 = dfpam #Log the values

corr = dfpam2.corr() #Compute the correlations
mask = np.triu(np.ones_like(corr, dtype=np.bool)) #Mask the top right

fig, ax = plt.subplots(1,1,figsize=(9,6),dpi=120)
sns.heatmap(round(corr,3), mask=mask, cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, annot=True, annot_kws={"fontsize":12}, cbar=False)
bottom, top = ax.get_ylim()
left, right = ax.get_xlim()
plt.ylim(bottom, top + 1)
plt.xlim(left, right - 1)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12, rotation=90)
plt.tick_params(axis='y', labelrotation = 90)

plt.show()

#%% Plot chosen pairs of parameter posteriors
#Indices for matching in plot
scat_first = [0,0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1, 2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3, 4,4,4,4,4,4, 5,5,5,5,5, 6,6,6,6, 7,7,7, 8,8, 9] #x axis parameter
scat_sec =   [1,2,3,4,5,6,7,8,9,10,2,3,4,5,6,7,8,9,10,3,4,5,6,7,8,9,10,4,5,6,7,8,9,10,5,6,7,8,9,10,6,7,8,9,10,7,8,9,10,8,9,10,9,10,10]
"""
"""
vamin = np.min(accepted_results)
vamax = np.max(accepted_results)
"""
f2, ax2 = plt.subplots(7,8,figsize=(14,16),dpi=120)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.35, hspace=0.5)

for i in range(55):
    ax = plt.subplot(7,8,i+1)   
    # if scat_first[i] != 4:
    #     plt.gca().set_xscale("log")
    # if scat_sec[i] in [0,1,2,3]:
    #     plt.gca().set_yscale("log")
    cm = plt.scatter(accepted_params[:,scat_first[i]],accepted_params[:,scat_sec[i]],s=2,c=accepted_results,vmin=vamin,vmax=vamax)
    ax.minorticks_off()
    plt.xlim(mins[scat_first[i]],maxs[scat_first[i]])
    plt.ylim(mins[scat_sec[i]],maxs[scat_sec[i]])
    plt.xlabel(labels[scat_first[i]],fontsize=13)
    plt.ylabel(labels[scat_sec[i]],fontsize=13)

"""
#  Choose which pairs are of interest
def Scat_getter(x):
    a = []
    for i in range(0,len(x)):
        for j in range(0,len(x)):
            if j <= i:
                continue
            else:
                if abs(x.iat[i,j]) > 0.35:
                    a.append([i,j,x.iat[i,j]])
                else:
                    continue
    return a

Scat_indices = Scat_getter(corr)
print(Scat_indices[0])

f3, ax3 = plt.subplots(5,3,figsize=(8,6),dpi=120)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.35, hspace=0.5)
   
for i in range(14):
    ax = plt.subplot(5,3,i+1)        
    plt.scatter(accepted_params[:,Scat_indices[i][0]],accepted_params[:,Scat_indices[i][1]],s=2)
    #cm = plt.scatter(accepted_params[:,Scat_indices[i][0]],accepted_params[:,Scat_indices[i][1]],s=2,c=accepted_results,vmin=vamin,vmax=vamax)
    ax.minorticks_off()
    #if scat_first[i] == 0:
    #    plt.xticks(np.arange(0.4, 0.6, 30))
    #ymin, ymax = ax.get_ylim()
    #ax.set_yticks(np.round(np.linspace(ymin, ymax, 3), 2))
    plt.xlabel(labelspost[Scat_indices[i][0]],fontsize=10)
    plt.ylabel(labelspost[Scat_indices[i][1]],fontsize=10)
    
plt.subplot(5,3,15)
plt.axis('off')
#Adds the colour bar      
# cb_ax = f2.add_axes([0.15, 0.1, 0.73, 0.03]) #x, y, width, height
# cbar = f2.colorbar(cm, cax=cb_ax, orientation='horizontal')

#plt.savefig('Para_scat_nut_del.png', bbox_inches='tight')

plt.show()

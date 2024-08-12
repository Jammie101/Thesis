import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import math
from scipy import integrate
import seaborn as sns
"""
Results = np.loadtxt('Pars_set_' + str(1) + '_iteration_0.txt')[:,0]
Params = np.loadtxt('Pars_set_' + str(1) + '_iteration_0.txt')[:,range(1,4)]
### Choose parameters to accept into posteriors ###
sorting = Results.argsort()
sorted_params = Params[sorting[::-1]]
one_percent = int(50000/100)"""
### Take the best 1% ###
abc = np.loadtxt('ABC-Pars_set_' + str(1) + '_iteration_0.txt')[:,range(1,4)]
ap1 = np.loadtxt('DummyPars_2.txt')[:,range(1,4)]
ap2 = np.loadtxt('DummyPars_4.txt')[:,range(1,4)]
ap3 = np.loadtxt('DummyPars_6.txt')[:,range(1,4)]
ap4 = np.loadtxt('DummyPars_8.txt')[:,range(1,4)]

accepted_params = np.loadtxt('DummyPars_10.txt')[:,range(1,4)]
accepted_results = np.loadtxt('DummyPars_10.txt')[:,0]
data = [101.38537462592477, 672.7424065515781, 3513.2824772888025, 7752.807304458312, 9584.127243820607, 10263.842848845608, 9544.767287157758]
sds = [4.539315530354904, 31.476372750410988, 244.69644902447075, 249.29735090151968, 457.6732658306751, 406.27655832766607, 331.93904230192203]
print(np.median(accepted_results))

numstep = 2401
t = np.linspace(0, 24, numstep)
t_lst = t.tolist()
tpts = np.linspace(0,24,7)
#Define a function to integrate the toxin ODE
def B_ana(t, K, B0, r):
         return (K*B0*np.exp(r*t))/(K+B0*(np.exp(r*t)-1))

### Run model simulations with the posterior distributions ###
if os.path.exists('simulations.txt'):
   sims_arr = np.loadtxt('simulations.txt')
else:
   sims = []
   for i in range(len(accepted_params)):
      print(i)
          
      B0,K,r = accepted_params[i,:].T    
      
      def B_ana(t, K, B0, r):
         return (K*B0*np.exp(r*t))/(K+B0*(np.exp(r*t)-1))

      model = B_ana(t, K, B0, r)
      sims.append(model)
   sims_arr = np.vstack((sims)).T
   
   np.savetxt('simulations.txt', sims_arr)

if os.path.exists('abcsimulations.txt'):
   abcsims_arr = np.loadtxt('abcsimulations.txt')
else:
   abcsims = []
   for i in range(len(abc)):
      print(i)
          
      B0,K,r = abc[i,:].T    
      
      def B_ana(t, K, B0, r):
         return (K*B0*np.exp(r*t))/(K+B0*(np.exp(r*t)-1))

      abcmodel = B_ana(t, K, B0, r)
      abcsims.append(abcmodel)
   abcsims_arr = np.vstack((abcsims)).T
   
   np.savetxt('abcsimulations.txt', abcsims_arr)

median = np.median(sims_arr, axis=1)
high = np.percentile(sims_arr, 97.5, axis=1)
low = np.percentile(sims_arr, 2.5, axis=1)

abcmedian = np.median(abcsims_arr, axis=1)
abchigh = np.percentile(abcsims_arr, 97.5, axis=1)
abclow = np.percentile(abcsims_arr, 2.5, axis=1)

### Plot the results ###
labels = ('$B_0$ CFU', '$K$ CFU', '$r$ hour$^{-1}$')
mins = [0,3,-2]
maxs = [4,7,2] 
numb = 1000
locs = [1,1,1]
"""
f1, ax1 = plt.subplots(1,3,figsize=(7.5,5),dpi=120)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.35, hspace=0.3)

for i in range(3):
   plt.subplot(1,3,i+1)
   prior = np.random.uniform(mins[i],maxs[i],numb)   
   sns.kdeplot(prior, shade=True, color='gray', alpha=0.6, label='Prior')
   if i in [2]:
      sns.kdeplot(accepted_params[:,i], shade=True, color='green', alpha=0.6, label=labels[i])
   else:
      sns.kdeplot(np.log10(accepted_params[:,i]), shade=True, color='green', alpha=0.6, label=labels[i])
   plt.grid()
   plt.legend(fontsize=12,loc='upper right')
   plt.ylabel('Density', fontsize=12)
   
plt.show()
"""
#plt.savefig('KDEs_nut_del.png',bbox_inches='tight')

fig = plt.subplots(1,2,figsize=(8,8),dpi=120)
plt.subplots_adjust(wspace=0.3)

plt.subplot(2,1,1)
plt.plot(t, abcmedian, label='Median',color='Orange')
plt.fill_between(t, abchigh, abclow, alpha=0.3, label='95% CI',color='Orange')
plt.scatter(tpts,  data, label='Data (mean)', color='Black',zorder=2)
plt.errorbar(tpts, data, yerr=np.squeeze(sds), capsize=3, ls='none', color='black')
plt.legend(loc=2, fontsize=12)
plt.ylabel('B(t)', fontsize=12)
#plt.ylim(-100,13500)
plt.yscale("log")

plt.subplot(2,1,2)
plt.plot(t, median, label='Median',color='Blue')
plt.fill_between(t, high, low, alpha=0.3, label='95% CI',color='Blue')
plt.scatter(tpts,  data, label='Data (mean)', color='Black',zorder=2)
plt.errorbar(tpts, data, yerr=np.squeeze(sds), capsize=3, ls='none', color='black')
plt.legend(loc=2, fontsize=12)
plt.ylabel('B(t)', fontsize=12)
plt.xlabel('Time (hours)', fontsize=12)
#plt.ylim(-100,13500)
plt.yscale("log")
#plt.ylim(0.5,5)
plt.show()
#plt.ylim(4,8)

#plt.savefig('Model_fit_nut_del.png',bbox_inches='tight')

"""dfpam = pd.DataFrame(accepted_params, columns=labels) #Create pandas dataframe
dfpam2 = np.log10(dfpam) #Log the values

corr = dfpam2.corr() #Compute the correlations
mask = np.triu(np.ones_like(corr, dtype=np.bool)) #Mask the top right

fig, ax = plt.subplots(1,1,figsize=(6,4),dpi=120)
sns.heatmap(corr, mask=mask, cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, annot=True, annot_kws={"fontsize":12}, cbar=False)
bottom, top = ax.get_ylim()
left, right = ax.get_xlim()
plt.ylim(bottom, top + 1)
plt.xlim(left, right - 1)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12, rotation=90)
plt.tick_params(axis='y', labelrotation = 90)

plt.show()
"""
#Scatter plots of pairs of parameter posteriors
scat_first = [0,0,1] #x axis parameter
scat_sec = [1,2,2] #y axis parameter
"""
f2, ax2 = plt.subplots(3,2,figsize=(8,6),dpi=120)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.35, hspace=0.5)
   
for i in range(15):
    ax = plt.subplot(5,3,i+1)        
    plt.gca().set_xscale("log")
    if scat_sec[i] != 5:
        plt.gca().set_yscale("log")
    plt.scatter(accepted_params[:,scat_first[i]],accepted_params[:,scat_sec[i]])
    ax.minorticks_off()
    #if scat_first[i] == 0:
    #    plt.xticks(np.arange(0.4, 0.6, 30))
    #ymin, ymax = ax.get_ylim()
    #ax.set_yticks(np.round(np.linspace(ymin, ymax, 3), 2))
    plt.xlabel(labels[scat_first[i]],fontsize=13)
    plt.ylabel(labels[scat_sec[i]],fontsize=13)"""
"""
vamin = np.min(accepted_results)
vamax = np.max(accepted_results)

f2, ax2 = plt.subplots(1,3,figsize=(9,15),dpi=120)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.35, hspace=0.5)

for i in range(3):
    ax = plt.subplot(1,3,i+1)   
    if scat_sec[i] != [2]:
        plt.gca().set_yscale("log")
    cm = plt.scatter(accepted_params[:,scat_first[i]],accepted_params[:,scat_sec[i]],s=2,c=accepted_results,vmin=vamin,vmax=vamax)
    ax.minorticks_off()
    if scat_first[i] != 4:
        plt.xlim(10**mins[scat_first[i]],10**maxs[scat_first[i]])
    elif scat_first ==4:
        plt.xlim(mins[scat_first[i]],maxs[scat_first[i]])
    if scat_sec[i] in [0,1,2,3]:
        plt.ylim(10**mins[scat_sec[i]],10**maxs[scat_sec[i]])
    if scat_sec[i] in [4,5]:
        plt.ylim(mins[scat_sec[i]],maxs[scat_sec[i]])
    plt.xlabel(labels[scat_first[i]],fontsize=13)
    plt.ylabel(labels[scat_sec[i]],fontsize=13)
  
#Adds the colour bar      
cb_ax = f2.add_axes([0.15, 0.1, 0.73, 0.03]) #x, y, width, height
cbar = f2.colorbar(cm, cax=cb_ax, orientation='horizontal')

#plt.savefig('Para_scat_nut_del.png', bbox_inches='tight')
plt.show()
#%%
"""
f2, ax2 = plt.subplots(2,3,figsize=(9,6),dpi=120)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.35, hspace=0.5)

for i in range(3):
    ax = plt.subplot(2,3,i+1)
    if scat_first[1] == 0:
        plt.xlim(10**0,10**4)
    if scat_first[i] == 1:
        plt.xlim(10**3,10**7)
    if scat_sec[i] == 1:
        plt.ylim(10**3,10**7)
    if scat_sec[i] == 2:
        plt.ylim(0,2)
    if scat_sec[i] != 2:
        plt.gca().set_yscale("log")
    plt.gca().set_xscale("log")
    plt.scatter(abc[:,scat_first[i]],abc[:,scat_sec[i]])
    ax.minorticks_off()
    """
    elif scat_first ==4:
        plt.xlim(mins[scat_first[i]],maxs[scat_first[i]])
    if scat_sec[i] in [0,1,2,3]:
        plt.ylim(10**mins[scat_sec[i]],10**maxs[scat_sec[i]])
    if scat_sec[i] in [4,5]:
        plt.ylim(mins[scat_sec[i]],maxs[scat_sec[i]])
    """
    plt.xlabel(labels[scat_first[i]],fontsize=13)
    plt.ylabel(labels[scat_sec[i]],fontsize=13)
for i in range(3):
    ax = plt.subplot(2,3,3+i+1)   
    if scat_first[1] == 0:
        plt.xlim(10**0,10**4)
    if scat_first[i] == 1:
        plt.xlim(10**3,10**7)
    if scat_sec[i] == 1:
        plt.ylim(10**3,10**7)
    if scat_sec[i] == 2:
        plt.ylim(0,2)
    plt.gca().set_xscale("log")
    if scat_sec[i] != [2]:
        plt.gca().set_yscale("log")
    plt.scatter(ap1[:,scat_first[i]],ap1[:,scat_sec[i]])
    plt.scatter(ap2[:,scat_first[i]],ap2[:,scat_sec[i]])
    plt.scatter(ap3[:,scat_first[i]],ap3[:,scat_sec[i]])
    plt.scatter(ap4[:,scat_first[i]],ap4[:,scat_sec[i]])
    plt.scatter(accepted_params[:,scat_first[i]],accepted_params[:,scat_sec[i]])
    ax.minorticks_off()
    plt.xlim(10**mins[scat_first[i]],10**maxs[scat_first[i]])
    if scat_sec[i] in [0,1]:
        plt.ylim(10**mins[scat_sec[i]],10**maxs[scat_sec[i]])
    if scat_sec[i] in [2]:
        plt.ylim(0,2)
        plt.gca().set_yscale("linear")
    plt.xlabel(labels[scat_first[i]],fontsize=13)
    plt.ylabel(labels[scat_sec[i]],fontsize=13)
   
#plt.savefig('Para_scat_nut_del.png', bbox_inches='tight')
plt.show()
# fig = plt.subplots(1,2,figsize=(10,3.5),dpi=120)
# plt.subplots_adjust(wspace=0.3)
"""
#%%
# plt.subplot(1,2,1)
# for i in range(len(accepted_params)):
#     plt.plot(t,bac_sims_arr[:,i],alpha=.3,zorder=1)
# plt.scatter([8,12,16,20], bac_data, label='Mean data', color='Black',zorder=2)
# plt.errorbar([8,12,16,20], bac_data, yerr=np.squeeze(bac_sd), capsize=3, ls='none', color='black')
# plt.legend(loc=2)
# plt.ylabel('Bacteria (CFU)')
# plt.xlabel('Time (secs)')

# plt.subplot(1,2,2)
# for i in range(len(accepted_params)):
#     plt.plot(t,pa_sims_arr[:,i],alpha=.3,zorder=1)
# plt.scatter([8,10,12,14,16,18,20], pa_data, label='Mean data', color='Black',zorder=2)
# plt.errorbar([8,10,12,14,16,18,20], pa_data, yerr=np.squeeze(pa_sd), capsize=3, ls='none', color='black')
# plt.legend(loc=2)
# plt.ylabel('Toxins (ng/ml)')
# plt.xlabel('Time (secs)')

# plt.savefig('Model_fit_sims.png')

###################################################################
# all_lst = []
# for i in range(20):
#     if i in [0,1,2,3,4,8,11,12,15]:
#        all_lst.append(np.loadtxt('Pars_set_' + str(i+1) + '_iteration_17.txt')[:,range(6)])
#     else:
#        all_lst.append(np.loadtxt('Pars_set_' + str(i+1) + '_iteration_16.txt')[:,range(6)])

# all_arr = np.vstack((all_lst))
# #all_arr = np.loadtxt('Pars_set_1_iteration_3.txt')[:,range(6)]

# test = all_arr[:, 0].argsort()
# sorts = all_arr[all_arr[:, 0].argsort()]
# cut = sorts[range(10),1:6]

# def dB_dt(B,t=0):
#    return np.array([lamb*B[0]*(1-(B[0]/k))])

# numstep = 161
# t = np.linspace(4, 20, numstep)

# B0 = bacteria.iloc[0,1]
# P0 = 0
# ICs = [B0,P0]

# bac_sims_srt = []
# pa_sims_srt = []
# for i in range(len(cut)):
#    print(i)
#    lamb, k, gamma, psi, dd = cut[i,:].T

#    B, infodict = integrate.odeint(dB_dt, B0, t, full_output = 1)
#    bac_model = B[:,0]
#    mfunc = interp1d(t, bac_model, bounds_error=False, fill_value="extrapolate")
   
#    def dP_dt(P,t,par):
#        k, gamma, psi = par.T
#        return np.array([gamma*mfunc(t)*(1-(mfunc(t)/k))-psi*P[0]])
   
#    delay = 4 + dd
#    delay_index = np.argmin(np.abs(t-delay))
#    parset2 = np.array((k,gamma,psi))
#    P = integrate.odeint(dP_dt, P0, t, args=(parset2,))  
#    leadzeros = np.repeat(0,delay_index)
#    P2 = np.hstack((leadzeros,np.squeeze(P)))
#    P3 = P2[range(numstep)]   

#    bac_sims_srt.append(B[:,0])
#    pa_sims_srt.append(P3)

# bac_sims_arr_srt = np.vstack((bac_sims_srt)).T
# pa_sims_arr_srt = np.vstack((pa_sims_srt)).T

# fig = plt.subplots(1,2,figsize=(10,3.5),dpi=120)
# plt.subplots_adjust(wspace=0.3)

# plt.subplot(1,2,1)
# for i in range(len(cut)):
#     plt.plot(t,bac_sims_arr_srt[:,i],alpha=.3,zorder=1)
# plt.scatter([8,12,16,20], bac_data, label='Mean data', color='Black',zorder=2)
# plt.errorbar([8,12,16,20], bac_data, yerr=np.squeeze(bac_sd), capsize=3, ls='none', color='black')
# plt.legend(loc=2)
# plt.ylabel('Bacteria (CFU)')
# plt.xlabel('Time (secs)')

# plt.subplot(1,2,2)
# for i in range(len(cut)):
#     plt.plot(t,pa_sims_arr_srt[:,i],alpha=.3,zorder=1)
# plt.scatter([8,10,12,14,16,18,20], pa_data, label='Mean data', color='Black',zorder=2)
# plt.errorbar([8,10,12,14,16,18,20], pa_data, yerr=np.squeeze(pa_sd), capsize=3, ls='none', color='black')
# plt.legend(loc=2)
# plt.ylabel('Toxins (ng/ml)')
# plt.xlabel('Time (secs)')

# plt.savefig('Model_fit_sims_best.png')
"""
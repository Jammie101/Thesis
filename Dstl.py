import numpy as np
import random
from scipy.integrate import odeint
import scipy.stats as st

import sys
index = sys.argv[1]

#index = 1
spore_data = np.loadtxt('C:\\Users\\Jamie\\Documents\\ABC-SMC\\Data\\Dstl Data\\spore_data.txt')
spore_sds = np.loadtxt('C:\\Users\\Jamie\\Documents\\ABC-SMC\\Data\\Dstl Data\\spore_sds.txt')

##### Bacteria data #####
bac_data = np.loadtxt('C:\\Users\\Jamie\\Documents\\ABC-SMC\\Data\\Dstl Data\\bac_data.txt')
bac_sds = np.loadtxt('C:\\Users\\Jamie\\Documents\\ABC-SMC\\Data\\Dstl Data\\bac_sds.txt')

##### PA data in micrograms #####

PA_data = np.loadtxt('C:\\Users\\Jamie\\Documents\\ABC-SMC\\Data\\Dstl Data\\pa_data.txt')
PA_sds = np.loadtxt('C:\\Users\\Jamie\\Documents\\ABC-SMC\\Data\\Dstl Data\\pa_sds.txt')

##### Define the time points of the data #####

spore_data_times = np.array([0, 1.5, 3.5, 4.25, 5, 6 ,7]) 

bac_data_times = np.array([0, 1.5, 3.5, 4.25, 5, 6 ,7, 16,18,20,22,24,40])
bl_data = 10**bac_data

pa_data_times = np.array([16, 18, 20, 22, 24, 40])

gran = 100 #100 timepoints per hour
t_max = 40
sim_times = np.linspace(0, t_max, t_max*gran +1)

t = sim_times 
#indexes of the data time points in the simulation timecourse
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
    spore_sim_full = R[:,0] + (1-f)*S0
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
      
    spore_sim = np.log10(spore_sim_full[spore_data_times[1:]]) + np.random.normal(0, spore_sds[1:])
    bac_sim = np.log10(NGB_sim_full[bac_data_times[1:]] + veg_sim_full[bac_data_times[1:]]) + np.random.normal(0, bac_sds[1:])
    PA_sim = PA_sim_full[pa_data_times] + np.random.normal(0, PA_sds)
    
    return spore_sim, bac_sim, PA_sim


#distance function
    
def d(x,xstar):
    return sum((x-xstar)**2)

def dl(x,xstar):
    return sum((np.log(x)-np.log(xstar))**2)

#               g, m, lamb, K, alph, beta,nu,e, tau1, tau2, f
prior_lowers = [-3, -3, -1, 5, -12, -7, -15, 0,  0,  0, 0]
prior_uppers = [1, 1,   1, 8,  -3,  0,  0,  1, 15, 24, 1]


#number of parameters
nump=11

#This function is just to generate the value of epsilon to use at the first iteration
#since it can be difficult to decide on a "reasonably large" value
def generate_eps1(n,r):
    print('generating initial epsilon')
    dists1 = np.empty((0))
    dists2 = np.empty((0))
    #n is the sample size and r is some number to multiply it by
    #e.g 1000 and 5
    accepted = 0
    while accepted < n*r:
        # Sample the parameters from uniform distributions
        paramstar  = []
        for i in range(7):
            paramstar.append(random.uniform(prior_lowers[i], prior_uppers[i]))
        for i in [7, 8, 9, 10]:
            paramstar.append(round(random.uniform(prior_lowers[i], prior_uppers[i]),2))
        gstar, mstar, lambstar, Kstar, alphastar, betastar, nustar, estar, tau1star, tau2star, fstar = paramstar
        #fstar = random.uniform(0,1)
        #simulate the model
        model=simulation(gstar, mstar, lambstar, Kstar, alphastar, betastar, nustar, estar, tau1star, tau2star, fstar)
        if ((model[0] > 0).sum() == model[0].size).astype(int)==1:#and ((model[1] >= 0).sum() == model[1].size).astype(int)==1 and ((model[2] >= 0).sum() == model[2].size).astype(int)==1:
            accepted+=1
            #compute the distance between model and data
            dista = d(spore_data[1:],model[0])
            distb = d(bac_data[1:],model[1])
            dist1 = dista + distb
            dist2 = d(PA_data,model[2])   
            dists1 = np.hstack((dists1, dist1))
            dists2 = np.hstack((dists2, dist2))
    epsilon1 = np.median(dists1) #Compute epsilon to use at the first iteration
    epsilon2 = np.median(dists2) #Compute epsilon to use at the first iteration
    return epsilon1, epsilon2

#This function is for the first iteration of ABC in which we sample from the priors
#and assign all particles equal weights
def iteration1(n, eps1, eps2):
   print('Running iteration: 0')
   dists1 = np.empty((0,1))
   dists2 = np.empty((0,1))
   params = np.empty((0,nump))
   #number accepted
   accepted = 0
   #runs is number of runs done
   runs = 0
   while accepted < n:
       runs+=1
       # Sample the parameters from uniform distributions
       paramstar  = []
       for i in range(7):
           paramstar.append(random.uniform(prior_lowers[i], prior_uppers[i]))
       for i in [7, 8, 9, 10]:
           paramstar.append(round(random.uniform(prior_lowers[i], prior_uppers[i]),2))
       gstar, mstar, lambstar, Kstar, alphastar, betastar, nustar, estar, tau1star, tau2star, fstar = paramstar
       #simulate the model
       model=simulation(gstar, mstar, lambstar, Kstar, alphastar, betastar, nustar, estar, tau1star, tau2star, fstar)
       if ((model[0] > 0).sum() == model[0].size).astype(int)==1: #and ((model[1] >= 0).sum() == model[1].size).astype(int)==1 and ((model[2] >= 0).sum() == model[2].size).astype(int)==1:
           dista = d(spore_data[1:],model[0])
           distb = d(bac_data[1:],model[1])
           dist1 = dista + distb
           dist2 = d(PA_data,model[2])   
           #If the distance is less than the epsilon value for this iteration 
           #keep the parameters
           if dist1 < eps1 and dist2 < eps2:
              accepted+=1
              dists1 = np.vstack((dists1, dist1))
              dists2 = np.vstack((dists2, dist2))
              params = np.vstack((params, np.array((gstar, mstar, lambstar, Kstar, alphastar, betastar, nustar, estar, tau1star, tau2star, fstar))))
#Give each accepted parameter set an equal weight
   weights = np.empty((0,1))
   for i in range(n):
      weights = np.vstack((weights,1/n))

   return [np.hstack((dists1, dists2)), np.hstack((dists1 + dists2, params, weights)), runs]

#This function is for all other iterations of the ABC where we now sample from the 
#previous iteration's posteriors
def other_iterations(n, eps1, eps2):

   #Since I am using a uniform kernel I am here computing the ranges of each of the
   #parameters, from the previous iteration
   ranges = []
   for i in range(nump): #number of params you have
      r = max(results[-1][:,i+1]) - min(results[-1][:,i+1])
      ranges.append(r)
   ranges = np.asarray(ranges)
   sigmas = 0.1*ranges #Take some fraction of these ranges to use to resample the parameters

   #To use when sampling the new parameters
   rows = [i for i in range(n)]
   #Upper and lower bound of each of the parameters, so that we don't end up
   #sampling outside of the original priors
   lower_bounds = [-3, -3, -1, 6, -12, -7, -15, 0,  0,  0, 0]
   upper_bounds = [1, 1,   1, 9,  -3,  0,  0,  1, 15, 24, 1]
   #the indexes for the ones you want to check the bound for
   b_indexes = [0, 1, 2, 3, 4, 5 , 6, 7, 8, 9, 10]
   #Some empty arrays for all the things we want to save
   dists1 = np.empty((0,1))
   dists2 = np.empty((0,1))
   params = np.empty((0,nump))
   weights = np.empty((0))

   accepted = 0
   runs = 0
   #While loop so that we accept n parameter sets
   while accepted < n:
      runs+=1
      check = 0
      #This while loop is to sample the parameters from the previous posteriors,
      #perturb them and check that they still lie within the initial prior ranges.
      while check < 1:
         row = np.random.choice(rows,1,p=results[-1][:,nump+1]) #Randomly choose from the posterior with the weights
         params_sample = results[-1][:,range(1,nump+1)][row]
         parameters = []
         for i in range(nump):
            lower = params_sample[0,i]-sigmas[i]
            upper = params_sample[0,i]+sigmas[i]
            parameter = np.random.uniform(lower, upper) #Sample using uniform kernel
            parameters.append(parameter)

         check_out = 0
         for i in b_indexes:
            if parameters[i] < lower_bounds[i] or parameters[i] > upper_bounds[i]:
               check_out = 1
         if check_out == 0:
            check+=1
            
      g = float(parameters[0])
      m = float(parameters[1])
      lamb = float(parameters[2])
      K = float(parameters[3])
      alpha = float(parameters[4])
      beta = float(parameters[5])
      nu = float(parameters[6])
      e = float(parameters[7])
      tau1 = round(float(parameters[8]), 2)
      tau2 = round(float(parameters[9]), 2)
      f = float(parameters[10]) 
      model=simulation(g, m, lamb, K, alpha, beta, nu, e, tau1, tau2, f)
      if ((model[0] > 0).sum() == model[0].size).astype(int)==1:# and ((model[1] >= 0).sum() == model[1].size).astype(int)==1 and ((model[2] >= 0).sum() == model[2].size).astype(int)==1:
          dista = d(spore_data[1:],model[0])
          distb = d(bac_data[1:],model[1])
          dist1 = dista + distb
          dist2 = d(PA_data,model[2]) 
          if dist1 < eps1 and dist2 < eps2:
             accepted+=1
             
             #Compute the denominator of the weights (sorry the code here is not amazing)
             denoms = []
             for j in range(n):
                previous_weight = results[-1][j,nump+1]
                previous_params = results[-1][j,1:nump+1]
                uppers = []
                lowers = []
                for i in range(nump):
                   uppers.append(previous_params[i] + sigmas[i])
                   lowers.append(previous_params[i] - sigmas[i])
                outside = 0
                for i in range(nump):
                   if parameters[i] < lowers[i] or parameters[i] > uppers[i]:
                      outside = 1                  
                if outside == 1:
                   denoms.append(0)
                else:
                   denoms.append(previous_weight*np.prod(1/(2*sigmas)))
             #Compute the final weight. Since I am using uniform priors, I know that the numerators
             #will cancel in the normalisation of the weights so I have just done 1/np.sum(demon_arr)
             #but if your priors aren't uniform you should replace the 1 with the prior densities
             weight = 1/np.sum(denoms)
             weights = np.hstack((weights,weight))
             
             dists1 = np.vstack((dists1, dist1))
             dists2 = np.vstack((dists2, dist2))
             params = np.vstack((params, np.array((g, m, lamb, K, alpha, beta, nu, e, tau1, tau2, f))))
             #priors_abc = np.vstack((priors_abc, prior_sample))
   #Normalise the weights
   normed_weights = weights/np.sum(weights)
   weights = np.reshape(normed_weights, (n,1))
            
   return [np.hstack((dists1, dists2)), np.hstack((dists1+dists2, params, weights)), runs]

#Having defined the functions, the lines below are to run the ABC SMC

#option1 = int(input("Posterior sample size = "))
#option2 = int(input("Number of iterations (if over 9 then colours will repeat on plots)= "))


#do 10 codes in parallel
sample_size = 200
num_iters = 20

#Generate fist value of epsilon
eps1, eps2 = generate_eps1(sample_size, 5)
print(eps1, eps2)


#don't save the first epsilons, only save the epilons that are used for the iterations that get saved, so that the numbers match
Epsilons1 = []
Epsilons2 = []

#Run first round of ABC
first_output = iteration1(sample_size, eps1, eps2)

eps1 = np.median(first_output[0][:,0])
eps2 = np.median(first_output[0][:,1])
print(eps1, eps2)

Epsilons1.append(eps1)
Epsilons2.append(eps2)

sep_dists = [first_output[0]]
results = [first_output[1]] #3 parts (distances, params, weights)
total_runs = [first_output[2]] #number of runs

#Run all other rounds of ABC
for k in range(num_iters):
   print('Running iteration: ' + str(k+1))
   
   runi = other_iterations(sample_size, eps1, eps2)
   
   eps1 = np.median(runi[0][:,0])
   eps2 = np.median(runi[0][:,1])
   
   #check if it has 'converged' and if so, use the previous epsilon again
   '''
   if Epsilons1[-1] - eps1 < 0.1:
       eps1 = Epsilons1[-1]
   '''
   
   '''
   if Epsilons2[-1] - eps2 < 10**-1:
        eps2 = Epsilons2[-1]
   '''
    
   Epsilons1.append(eps1)
   Epsilons2.append(eps2)
   print(eps1, eps2)
   
   sep_dists.append(runi[0])
   results.append(runi[1])
   total_runs.append(runi[2])
   
   
   np.savetxt('results_set' + str(index) + '.txt',results[-1])
   np.savetxt('Epsilons1_set' + str(index) + '.txt', Epsilons1)
   np.savetxt('Epsilons2_set' + str(index) + '.txt', Epsilons2)
   
   params=results[-1][:,range(1,nump+1)]
   dists=results[-1][:,0]
    
   sorting = dists.argsort()
    
   sorted_dists = dists[sorting[::-1]]
   sorted_params = params[sorting[::-1]]
   sorted_sep_dists = sep_dists[-1][sorting[::-1]]
   
   np.savetxt('separated_dists_iter'+str(k+1)+'_set' + str(index) + '.txt', sorted_sep_dists)
   np.savetxt('dists_sorted_iter'+str(k+1)+'_set' + str(index) + '.txt', sorted_dists)
   np.savetxt('posterior_sorted_iter'+str(k+1)+'_set' + str(index) + '.txt', sorted_params)
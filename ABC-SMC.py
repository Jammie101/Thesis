import numpy as np
import pandas as pd
from scipy import integrate
index = 1
data = np.reshape(np.array([101.38537462592477, 672.7424065515781, 3513.2824772888025, 7752.807304458312, 9284.127243820607, 10263.842848845608, 9544.767287157758]),(7,1))
sds = np.reshape(np.array([4.539315530354904, 31.476372750410988, 244.69644902447075, 249.29735090151968, 457.6732658306751, 406.27655832766607, 331.93904230192203]),(7,1))
#%%
B0 = 100
K = 1E4
r = 0.5
#%%
numstep = 2401
t = np.linspace(0, 24, numstep)
tps = [0,400,800,1200,1600,2000,2400]
def B_ana(t, K, B0, r):
         return (K*B0*np.exp(r*t))/(K+B0*(np.exp(r*t)-1))
     
        
print(B_ana(t, K, B0, r))
print(B_ana(t, K, B0, r)[tps])
model = np.reshape(B_ana(t, K, B0, r)[tps],(7,1))
print((model-data)/sds)
#Function for the first iteration of the ABC-SMC where the parameters are sampled from the prior distributions.
def first_iteration(N,ep):
   print('Running iteration: 0')
   epsilon = ep
   accepted_params = np.empty((0,3))
   results = np.empty((0))
   number = 0 #Counter for the population. 
   truns = 0 
   while number < N: #Run the loop until N parameter sets are accepted.
      print(number)
      truns+=1
      #Draw the parameters from the prior distributions.
      B0 = pow(10,np.random.uniform(0,4))
      K = pow(10,np.random.uniform(3,7))
      r = np.random.uniform(0,2)
      parset1 = np.array([B0,K,r])
      
      sim_model = np.reshape(B_ana(t, K, B0, r)[tps],(7,1))
      sim_final = sim_model + np.random.normal(-sds,sds)
      d1 = np.sum(pow(np.log10(sim_model)-np.log10(data),2))
      diff = np.sqrt(d1)
      #If the distance if less than the first value of epsilon, append the parameters to the accepted parameters array,
      #and add one to the parameter set counter.
      if diff < epsilon:
         number+=1
         results = np.hstack((results,diff))
         accepted_params = np.vstack((accepted_params, parset1))
            
   #Compute the weight for each accepted parameter set (at iteration 1, the parameter sets are equally weighted).
   weights = np.empty((0,1))
   for i in range(len(accepted_params)):
      weights = np.vstack((weights,1/len(accepted_params)))
   
   #Print information about the first run of the ABC.
   print('Acceptance rate for iteration 0: ' + str(N*100/truns))
   print('Epsilon = ' + str(epsilon))
   print('Total runs = ' + str(truns))
   #Return the results (distances, accepted parameters, weights, total number of runs).
   return [np.hstack((np.reshape(results,(len(accepted_params),1)),accepted_params,weights)),truns]

#Function for the other iterations of the ABC-SMC where the parameters are sampled from the previous posteriors.
def other_iterations(N,it):
   print('Running iteration: ' + str(it+1))
   
   epsilon = np.median(ABC_runs[it][:,0])
   p_list = [i for i in range(N)]
   
   #Upper and lower bounds for the uniform distributions of the priors for each parameter.
   lower_bounds = [0, 3, 0]
   upper_bounds = [4,7,2] 
   
   #Compute uniform areas to sample within in order to perturb the parameters.
   ranges = []
   for i in range(3):
      if i in [2]:
          r1 = np.max(ABC_runs[it][:,i+1]) - np.min(ABC_runs[it][:,i+1])
      else:
         r1 = np.max(np.log10(ABC_runs[it][:,i+1])) - np.min(np.log10(ABC_runs[it][:,i+1]))
      ranges.append(r1)
   ranges_arr = np.asarray(ranges)
   sigma = 0.2*ranges_arr

   priors = np.empty((0,3))
   accepted_params = np.empty((0,3))
   results = np.empty((0))
   weights = np.empty((0))

   number = 0 #Counter for the population. 
   truns = 0 
   while number < N: #Run the loop until N parameter sets are accepted.
      #print(number)
      truns+=1
      check = 0
      while check < 1:
         choice = np.random.choice(p_list,1,p=ABC_runs[it][:,4]) #Choose a random parameter set from previous iterations posteriors.
         prior_sample = ABC_runs[it][:,range(1,4)][choice]
         #Generate new parameters through perturbation.
         parameters = []
         for i in range(3):
            if i in [2]:
               lower = prior_sample[0,i]-sigma[i]
               upper = prior_sample[0,i]+sigma[i]
            else:
               lower = np.log10(prior_sample[0,i])-sigma[i]
               upper = np.log10(prior_sample[0,i])+sigma[i]
            parameter = np.random.uniform(lower, upper)
            if i in [2]:
               parameters.append(parameter)
            else:
               parameters.append(pow(10,parameter))
         #Check that the new parameters are feasible given the priors.
         check_out = 0
         for ik in range(3):
            if ik in [2]:
               if parameters[ik] < lower_bounds[ik] or parameters[ik] > upper_bounds[ik]:
                  check_out = 1
            else:
               if parameters[ik] < pow(10,lower_bounds[ik]) or parameters[ik] > pow(10,upper_bounds[ik]):
                  check_out = 1
         if check_out == 0:
            check+=1

      B0 = parameters[0]
      K = parameters[1]
      r = parameters[2]
      parset1 = np.array([B0,K,r])
      
      sim_model = np.reshape(B_ana(t, K, B0, r)[tps],(7,1))
      sim_final = sim_model + np.random.normal(-sds,sds)
      d1 = np.sum(pow(np.log10(sim_model)-np.log10(data),2))
      diff = np.sqrt(d1)
      #If the distance is less than epsilon, append distance and parameters to their respective lists and compute the weight
      #for the accepted parameter set.
      if diff < epsilon:
         number+=1         
         denom_arr = []
         for j in range(N):
            weight = ABC_runs[it][j,4]
            params_row = ABC_runs[it][j,1:4]
            boxs_up = []
            boxs_low = []
            for i in range(3):
               if i in [2]:
                  boxs_up.append(params_row[i] + sigma[i])
                  boxs_low.append(params_row[i] - sigma[i])
               else:
                  boxs_up.append(np.log10(params_row[i]) + sigma[i])
                  boxs_low.append(np.log10(params_row[i]) - sigma[i])
            outside = 0
            for i in range(3):
               if i in [2]:
                  if parameters[i] < boxs_low[i] or parameters[i] > boxs_up[i]:
                     outside = 1
               else:
                  if np.log10(parameters[i]) < boxs_low[i] or np.log10(parameters[i]) > boxs_up[i]:
                     outside = 1                  
            if outside == 1:
               denom_arr.append(0)
            else:
               denom_arr.append(weight*np.prod(1/(2*sigma)))
         weight_param = 1/np.sum(denom_arr)
         
         weights = np.hstack((weights,weight_param))
         results = np.hstack((results, diff))
         accepted_params = np.vstack((accepted_params, parset1))
         priors = np.vstack((priors, prior_sample))

   #Normalise the weights
   weights_2 = weights/np.sum(weights)
   weights_3 = np.reshape(weights_2, (len(weights_2),1))
   
   #Print information about the first run of the ABC.
   print('Acceptance rate for iteration ' + str(it+1) + ': ' + str(N*100/truns))
   print('Epsilon = ' + str(epsilon))
   print('Total runs = ' + str(truns))
   #Return the results (distances, accepted parameters, weights, total number of runs).
   return [np.hstack((np.reshape(results,(len(accepted_params),1)),accepted_params,weights_3)),truns]

#Sample size for the ABC-SMC
N = 500
#Run the first iteration with a sufficiently large value of epsilon.
first = first_iteration(N,10)
ABC_runs = []
ABC_runs.append(first[0])
np.savetxt('Pars_set_' + str(index) + '_iteration_0.txt', ABC_runs[0])
#Run the sucessive iterations of the ABC.

for it in range(10):   
   run = other_iterations(N,it)
   ABC_runs.append(run[0])
   np.savetxt('DummyPars_'+str(it+1)+'.txt', ABC_runs[it+1])    
#index = sys.argv[1]
#Data reshaped to be the same shape as the model output

#A16R (Baca from Model-Calibration) data for Zai paper
"""
K = 1E4
r = 0.5
B0 = 1E2
t = np.linspace(0,24,2401)
def B_ana(t):
         return (K*B0*np.exp(r*t))/(K+B0*(np.exp(r*t)-1))

print(B_ana(t))
print(B_ana(t)[tps])
r1 = []
r2 = []
r3 = []
r4 = []
r5 = []

for i in range(len(m)):
    r1.append(m[i] + np.random.uniform(-0.1*m[i],0.1*m[i]))
    r2.append(m[i] + np.random.uniform(-0.1*m[i],0.1*m[i]))
    r3.append(m[i] + np.random.uniform(-0.1*m[i],0.1*m[i]))
    r4.append(m[i] + np.random.uniform(-0.1*m[i],0.1*m[i]))
    r5.append(m[i] + np.random.uniform(-0.1*m[i],0.1*m[i]))

print(r1,r2,r3,r4,r5)
data = []
for i in range(len(m)):
    data.append((r1[i]+r2[i]+r3[i]+r4[i]+r5[i])/5)
    
print(data)

r1 = [100.12644027370268, 625.2839534263833, 3712.761708727485, 7684.240678493392, 9288.044408092475, 10771.939737581053, 9921.772893821899]
"""
r2 = [109.83873999772251, 646.7339998793253, 3894.431836478793, 7680.918666625899, 9758.440893126399, 9988.279048821845, 9241.960034968197]
r3 = [100.97295576021756, 704.6085371513591, 3293.5005418592323, 7403.78826484954, 9339.49654329753, 9948.905158872174, 9965.030912411601]
r4 = [99.84026831060966, 685.0484317806505, 3311.5787641868515, 8169.69214062167, 9133.961108663334, 10745.85369368039, 9202.078609869173]
r5 = [96.14846878737141, 702.0371105201731, 3354.1395351916503, 7825.396771701051, 10400.6932659233, 9864.236605272577, 9392.993984717921]

print(np.round(sds,2))

"""
for i in range(len(data)):
    varlist = []
    varlist.append(r1[i])
    varlist.append(r2[i])
    varlist.append(r3[i])
    varlist.append(r4[i])
    varlist.append(r5[i])
    print(varlist)
    v = np.var(varlist)
    var.append(v)
print(var)

sds = []
for i in range(len(data)):
    sds.append(np.sqrt(var[i]))
print(sds)
"""
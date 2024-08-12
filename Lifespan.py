import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

#fixed parameters
lamba = 0.8
gamma1 = 1
mu1 = 1


#variable parameters starting values
mu2 = 0.1*mu1
gamma2 = 0.1*gamma1
nu2dv = 0
nu1 = 0.675
nu2 = 0
N = 100
fp = [0.675]*N
mu2dv = 0.1
i = 1
j=0

while i < 4: 
    # generate 2 2d grids for the x & y bounds
    y, x = np.meshgrid(np.linspace(0, 5, N), np.linspace(0, 5, N))
    
    #x = nu_12
    #y = nu_21
    
    a = (1 - (x*y)/((mu1+x)*(mu2+y)))
    z = (1/a)*(1/(mu1+x))*((x/(mu2+y))+1)
    
    # x and y are bounds, so z should be the value *inside* those bounds.
    # Therefore, remove the last value from the z array.
    z = z[:-1, :-1]
    z_min, z_max = -np.abs(z).max(), np.abs(z).max()
    
    plt.subplot(1,3,j+i)
    plt.pcolor(x, y, z, cmap='RdYlBu_r',norm=LogNorm(vmin=0.1,vmax = 10))
    plt.colorbar(extend='max')
    #plt.plot(x,fp,"green", markersize=3)
    plt.ylabel("$\\nu_{}$".format({21}),fontsize=14, rotation=0)
    plt.xlabel("$\\nu_{}$".format({12}),fontsize=14)
    plt.axis([x.min(), x.max(), y.min(), y.max()])
    plt.title("$\mu_2 = {}$".format(mu2dv),fontsize=14)
    i+=1
    mu2dv *= 10
    print(mu2)
    mu2 *= 10

plt.suptitle("$\mathbb{E}[T_1]$", fontsize=14   )    
plt.show()
# Estimate the parameters of the Gaussian Process using MCMC algorithms

using Gadfly, GaussianProcesses

# Training data
n=10                 #number of training points
x = 2π * rand(n)              
y = sin(x) + 0.05*randn(n)

#Select mean and covariance function
mZero = MeanZero()                   #Zero mean function
kern = SE(0.0,0.0)                   #Sqaured exponential kernel (note that hyperparameters are on the log scale)

gp = GP(x,y,mZero,kern,-1.0)      #Fit the GP, where -1.0 is the log Gaussian noise
plot(gp)                          #Plot the GP

start = [-1.0,0.0,0.0,0.0]  #choose a starting value for the MCMC algorithm
mcrange = Lora.BasicMCRange(nsteps=50000, burnin=10000) #specify the number of iterations and burnin length


#Compare available samplers
sampler = Lora.MH(ones(4))              #MH sampler
MCMCout1=mcmc(gp,start,sampler,mcrange)

sampler = Lora.MALA(0.3)                #MALA sampler
MCMCout2=mcmc(gp,start,sampler,mcrange)

sampler = Lora.HMC(10,0.05)             #HMC sampler
MCMCout3=mcmc(gp,start,sampler,mcrange)

plot(layer(x=MCMCout1[3,:],Geom.density),
     layer(x=MCMCout2[3,:],Geom.density,Theme(default_color=colorant"red")),
     layer(x=MCMCout3[3,:],Geom.density,Theme(default_color=colorant"green")))

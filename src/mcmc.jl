@doc """
# Description
A function for running a variety of MCMC algorithms for estimating the GP hyperparameters. This function uses the MCMC algorithms provided by the Lore packages and the user is referred to this package for further details.

# Arguments:
* `gp::GP`: Predefined Gaussian process type
""" ->

function mcmc(gp::GP,start::Vector{Float64},sampler::Lora.MCSampler,mcrange::Lora.BasicMCRange; noise::Bool=true, mean::Bool=true, kern::Bool=true)
    
    function mll(hyp::Vector{Float64})  #log-target
        set_params!(gp, hyp; noise=noise, mean=mean, kern=kern)
        update_mll!(gp)
        return gp.mLL
    end
    
    function dmll(hyp::Vector{Float64}) #gradient of the log-target
        set_params!(gp, hyp; noise=noise, mean=mean, kern=kern)
        update_mll_and_dmll!(gp; noise=noise, mean=mean, kern=kern)
        return gp.dmLL
    end
        starting = Dict(:p=>start)
    q = BasicContMuvParameter(:p, logtarget=mll,gradlogtarget=dmll) 
    model = likelihood_model(q, false)                               #set-up the model
    tune = VanillaMCTuner(period=mcrange.burnin)                     #set length of tuning (default to burnin length)
    job = BasicMCJob(model, sampler, mcrange, starting,tuner=tune)   #set-up MCMC job
    print(job)                                                       
    run(job)
    chain = Lora.output(job)
    return chain.value
end    
    


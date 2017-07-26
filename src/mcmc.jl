@doc """
    # Description
    A function for running a variety of MCMC algorithms for estimating the GP hyperparameters. This function uses the MCMC algorithms provided by the Mambda package and the user is referred to this package for further details.

    # Arguments:
    * `gp::GPBase`: Predefined Gaussian process type
    * `nIters::Int64`: number of MCMC iterations
    * `burnin::Int64`: length of burnin
    * `ε::Float64`: step-size parameters
    * `init::Vector{Float64}`: Select a starting value, default is taken as current GP parameters
    """ ->
function mcmc(gp::GPBase,
              nIters::Int64,
              burnin::Int64,
              ε::Float64,
              init::Vector{Float64}=get_params(gp))

    ## Log-transformed Posterior and Gradient Vector
    logf = function(hyp::DenseVector)
        set_params!(gp, hyp)
        update_target!(gp)
        return gp.target
    end

    ## Log-transformed Posterior and Gradient Vector
    logfgrad = function(hyp::DenseVector)
        set_params!(gp, hyp)
        update_target_and_dtarget!(gp)
        return gp.target, gp.dtarget
    end
        
   #     sim = Chains(nIters, length(get_params(gp)), start = (burnin + 1))
        out = Array{Float64}(nIters, length(get_params(gp)))
        #theta = NUTSVariate(init, logfgrad)
        theta = MALAVariate(init, ε, logfgrad)
        for i in 1:nIters
            sample!(theta)
            out[i, :] = theta;
        end
        set_params!(gp,init)      #reset the parameters stored in the GP to their original values
        return out[(burnin+1):end,:]
    end    
    


    

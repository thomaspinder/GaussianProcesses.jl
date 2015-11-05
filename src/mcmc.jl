@doc """
# Description
A function to sample from the posterior distribution of the GP hyperparameters. This function uses the NUTS sampler (code taken from Mamba package) which is an automatically tuned HMC sampler and as such utilizes the gradients of the log-posterior.

    # Arguments:
    * `gp::GP`: Predefined Gaussian process type
        * `numIter::Int64`: Number of MCMC iterations
        * `burnin::Int64`: Number of initial MCMC iterations that are treated as burnin
        """ ->
function mcmc(gp::GP, numIter::Int64, burnin::Int64)

    #Log posterior
    function fx(theta::DenseVector)
        set_params!(gp, theta; mean=false, noise=false)
        update_mll_prior!(gp)
        logPost = gp.mLL -sum(0.5*(theta.^2)/10)
        return logPost
    end

    sim = Array(Float64,numIter,length(get_params(gp)))
    lPost = Array(Float64,numIter)
    theta = AMWGVariate(get_params(gp))   #theta = NUTSVariate(get_params(gp)) 
    #epsilon = nutsepsilon(theta, fx)
    sigma = ones(length(get_params(gp)))
    for i in 1:numIter
        amwg!(theta, sigma, fx, adapt = (i <= burnin)) #nuts!(theta, epsilon, fx, adapt = (i <= burnin))
        lPost[i] = gp.mLL -sum(0.5*(theta.^2)/10)
        sim[i,:] = collect(theta)
    end
    return sim[(burnin+1):end,:], lPost[(burnin+1):end]
end




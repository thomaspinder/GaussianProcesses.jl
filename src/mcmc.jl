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
        set_params!(gp, theta)
        update_mll_and_dmll!(gp)
        logPost = gp.mLL -sum(0.5*theta.^2)
        return logPost
    end

    sim = Array(Float64,numIter,length(get_params(gp)))
    theta = AMWGVariate(get_params(gp))  #        theta = NUTSVariate(get_params(gp))
    #epsilon = nutsepsilon(theta, fx)
    sigma = ones(length(get_params(gp)))
    for i in 1:numIter
        amwg!(theta, sigma, fx, adapt = (i <= burnin))  #nuts!(theta, epsilon, fx, adapt = (i <= burnin))
        sim[i,:] = collect(theta)
    end
    return sim[(burnin+1):end,:]
end




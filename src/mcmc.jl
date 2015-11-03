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
    fx = function(theta::DenseVector)
        set_params!(gp, theta)
        update_mll_and_dmll_prior!(gp::GP)
        logPost = gp.mLL
        grad = gp.dmLL
        logPost, grad
    end

        sim = Array(Float64,numIter, 1+num_params(gp.k))
        theta = NUTSVariate(get_params(gp))
        epsilon = nutsepsilon(theta, fx)
        for i in 1:numIter
            nuts!(theta, epsilon, fx, adapt = (i <= burnin))
        end
        sim
    end




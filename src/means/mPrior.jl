# Prior for the mean function

@doc """
# Description
Constructor for the prior mean function. We assume a Gaussian prior β ∼ N(b,B) as in Rasmussen and Williams (2006)

# Arguments:
* `b::Vector{Float64}`: A dxD length vector of prior means, where d is the degree of the polynomial and D is the dimension of the input space
* `B::Matrix{Float64}`: A (dxD) x (dXD) matrix representing the priorcovariance, where d is the degree of the polynomial and D is the dimension of the input space
""" ->
type MeanPrior <: Mean
    dim::Int64            # Dimension of the input space
    deg::Int64            # Degree of the polynomial
    MeanPrior(dim::Int64,deg::Int64) = new(dim,deg+1)
end

function meanf(mPrior::MeanPrior,x::Matrix{Float64})
    dim, nobsv = size(x)
    dim == mPrior.dim || throw(ArgumentError("Observations and mean function have inconsistent dimensions"))
    H = Array(Float64,mPrior.deg,nobsv)
    for i in 1:nobsv
        for j in 1:(mPrior.deg)
            H[j,i] = sum(x[:, i].^(j-1))
        end    
    end
    return H
end

get_params(mPrior::MeanPrior) = []
num_params(mPrior::MeanPrior) = 0

function set_params!(mPrior::MeanPrior, hyp::Vector{Float64})
    length(hyp) == 0 || throw(ArgumentError("Using this prior for the mean function does not require parameter optimization"))
end

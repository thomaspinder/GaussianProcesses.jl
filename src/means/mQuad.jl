# Quadratic mean function where hyperparameters are marginalised

@doc """
# Description
Constructor for the prior mean function. We assume a Gaussian prior β ∼ N(b,B) as in Rasmussen and Williams (2006)

# Arguments:
* `b::Vector{Float64}`: A dxD length vector of prior means, where d is the degree of the polynomial and D is the dimension of the input space
* `B::Matrix{Float64}`: A (dxD) x (dXD) matrix representing the priorcovariance, where d is the degree of the polynomial and D is the dimension of the input space
""" ->
type MeanQuad <: Mean
    dim::Int64            # Dimension of the input space
    MeanQuad(dim::Int64) = new(dim)
end

function meanf(mQuad::MeanQuad,x::Matrix{Float64})
    dim, nobsv = size(x)
    dim == mQuad.dim || throw(ArgumentError("Observations and mean function have inconsistent dimensions"))
    H = Array(Float64,3,nobsv)
    H[1,:] = ones(nobsv)
    H[2,:] = sum(x,1)
    H[3,:] = diag(x'x)
    return H
end

get_params(mQuad::MeanQuad) = []
num_params(mQuad::MeanQuad) = 0

function set_params!(mQuad::MeanQuad, hyp::Vector{Float64})
    length(hyp) == 0 || throw(ArgumentError("Using this prior for the mean function does not require parameter optimization"))
end

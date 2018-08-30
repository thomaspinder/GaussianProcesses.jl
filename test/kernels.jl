using GaussianProcesses, Test
using LinearAlgebra, Statistics, Calculus

@testset "Kernels" begin
    d, n, n2 = 2, 10, 5
    ll = rand(d)
    X = randn(d, n)
    X2 = randn(d, n2)
    y = randn(n)

    kernels = [# Isotropic kernels
               SEIso(1.0, 1.0), Mat12Iso(1.0,1.0), Mat32Iso(1.0,1.0), Mat52Iso(1.0,1.0),
               RQIso(1.0, 1.0, 1.0), Periodic(1.0, 1.0, 2π),
               # Non-isotropic
               Lin(1.0), Poly(0.0, 0.0, 2), Noise(1.0),
               # Constant kernel
               Const(1.0),
               # ARD kernels
               SEArd(ll, 1.0), Mat12Ard(ll, 1.0), Mat32Ard(ll, 1.0), Mat52Ard(ll, 1.0),
               RQArd(ll, 0.0, 2.0), LinArd(ll),
               # Composite kernels
               SEIso(1.0, 1.0) + Mat12Iso(1.0, 1.0),
               (SEIso(1.0, 1.0) + Mat12Iso(1.0, 1.0)) + Lin(1.0),
               SEIso(1.0, 1.0) * Mat12Iso(1.0, 1.0),
               (SEIso(1.0, 1.0) * Mat12Iso(1.0, 1.0)) * Lin(1.0),
               # Fixed Kernel
               fix(RQIso(1.0, 1.0, 1.0), :lσ), fix(RQIso(1.0, 1.0, 1.0)),
               # Sum and Product and Fix
               SEIso(1.0, 1.0) * Mat12Iso(1.0, 1.0) +
               Lin(1.0) * fix(RQIso(1.0, 1.0, 1.0), :lσ)]

    @testset "Kernel $(typeof(kernel))" for kernel in kernels
        println("\tTesting kernel ", typeof(kernel), "...")
        @test length(GaussianProcesses.get_param_names(kernel)) ==
            length(GaussianProcesses.get_params(kernel)) ==
            GaussianProcesses.num_params(kernel)

        @testset for masked in (true, false)
            if masked
                # This is a bit of a hack, to remove one parameter
                # from an ARD kernel to make it have the right number
                # of parameters when masked
                if isa(kernel, LinArd) || isa(kernel, GaussianProcesses.StationaryARD)
                    par = GaussianProcesses.get_params(kernel)[2:end]
                    k_masked = typeof(kernel)([par[1]], par[2:end]...)
                    kern = Masked(k_masked, [1])
                else
                    kern = Masked(kernel, [1])
                end
            else
                kern = kernel
            end

            @testset "Variance" begin
                spec = cov(kern, X)
                gen = invoke(cov, Tuple{Kernel, Matrix{Float64}}, kern, X)
                @test spec ≈ gen
                i, j = rand(1:n), rand(1:n)
                @test spec[i, j] ≈ cov(kern, X[:, i], X[:, j])
                cK = zeros(n, n)
                GaussianProcesses.addcov!(cK, kern, X)
                @test spec ≈ cK
                fill!(cK, 0)
                kdata = GaussianProcesses.KernelData(kern, X)
                GaussianProcesses.addcov!(cK, kern, X, kdata)
                @test spec ≈ cK
                fill!(cK, 1)
                GaussianProcesses.multcov!(cK, kern, X)
                @test spec ≈ cK
                fill!(cK, 1)
                GaussianProcesses.multcov!(cK, kern, X, kdata)
                @test spec ≈ cK
                key = GaussianProcesses.kernel_data_key(kern, X)
                @test typeof(key) == String
                # check we've overwritten the default if necessary
                if typeof(kdata) != GaussianProcesses.EmptyData
                    @test key != "EmptyData"
                end
            end

            @testset "Covariance" begin
                spec = cov(kern, X, X2)
                i, j = rand(1:n), rand(1:n2)
                @test spec[i,j] ≈ cov(kern, X[:, i], X2[:, j])
                cK = zeros(n, n2)
                GaussianProcesses.addcov!(cK, kern, X, X2)
                @test spec ≈ cK
                fill!(cK, 1)
                GaussianProcesses.multcov!(cK, kern, X, X2)
                @test spec ≈ cK
            end

            @testset "Gradient" begin
                nparams = GaussianProcesses.num_params(kern)
                init_params = GaussianProcesses.get_params(kern)
                data = GaussianProcesses.KernelData(kern, X)
                stack1 = Array{Float64}(undef, n, n, nparams)
                stack2 = Array{Float64}(undef, n, n, nparams)

                GaussianProcesses.grad_stack!(stack1, kern, X, data)
                invoke(GaussianProcesses.grad_stack!,
                       Tuple{AbstractArray, Kernel, Matrix{Float64},
                             GaussianProcesses.EmptyData},
                       stack2, kern, X, GaussianProcesses.EmptyData())
                @test stack1 ≈ stack2

                theor_grad = vec(sum(stack1; dims=[1,2]))
                numer_grad = Calculus.gradient(init_params) do params
                    set_params!(kern, params)
                    sum(cov(kern, X))
                end
                @test theor_grad ≈ numer_grad rtol=1e-1 atol=1e-2
            end

            @testset "dtarget" begin
                gp = GPE(X, y, MeanConst(0.0), kern, -3.0)
                init_params = GaussianProcesses.get_params(gp)
                GaussianProcesses.update_target_and_dtarget!(gp)
                theor_grad = copy(gp.dtarget)
                numer_grad = Calculus.gradient(init_params) do params
                    set_params!(gp, params)
                    update_target!(gp)
                    gp.target
                end
                @test theor_grad ≈ numer_grad rtol=1e-3 atol=1e-3
            end
        end
    end
end

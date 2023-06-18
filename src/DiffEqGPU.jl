"""
$(DocStringExtensions.README)
"""
module DiffEqGPU

using DocStringExtensions
using KernelAbstractions
import KernelAbstractions: get_backend, allocate
using SciMLBase, DiffEqBase, LinearAlgebra, Distributed
using ForwardDiff
import ChainRulesCore
import ChainRulesCore: NoTangent
using RecursiveArrayTools
import ZygoteRules
import Base.Threads
using LinearSolve
#For gpu_tsit5
using Adapt, SimpleDiffEq, StaticArrays
using Parameters, MuladdMacro
using Random
using Setfield
using ForwardDiff

"""
Wrapper for modifying parameters to contain additional data. Useful for simulating
trajectories with different time span values.
"""
struct ParamWrapper{P, T}
    params::P
    data::T
end

function Adapt.adapt_structure(to, ps::ParamWrapper{P, T}) where {P, T}
    ParamWrapper(adapt(to, ps.params),
        adapt(to, ps.data))
end

@kernel function gpu_kernel(f, du, @Const(u),
    @Const(params::AbstractArray{ParamWrapper{P, T}}),
    @Const(t)) where {P, T}
    i = @index(Global, Linear)
    @inbounds p = params[i].params
    @inbounds tspan = params[i].data
    @views @inbounds f(du[:, i], u[:, i], p, t)
    @inbounds for j in 1:size(du, 1)
        du[j, i] = du[j, i] * (tspan[2] - tspan[1])
    end
end

@kernel function gpu_kernel_oop(f, du, @Const(u),
    @Const(params::AbstractArray{ParamWrapper{P, T}}),
    @Const(t)) where {P, T}
    i = @index(Global, Linear)
    @inbounds p = params[i].params
    @inbounds tspan = params[i].data

    @views @inbounds x = f(u[:, i], p, t)
    @inbounds for j in 1:size(du, 1)
        du[j, i] = x[j] * (tspan[2] - tspan[1])
    end
end

@kernel function gpu_kernel(f, du, @Const(u), @Const(p), @Const(t))
    i = @index(Global, Linear)
    if eltype(p) <: Number
        @views @inbounds f(du[:, i], u[:, i], p[:, i], t)
    else
        @views @inbounds f(du[:, i], u[:, i], p[i], t)
    end
end

@kernel function gpu_kernel_oop(f, du, @Const(u), @Const(p), @Const(t))
    i = @index(Global, Linear)
    if eltype(p) <: Number
        @views @inbounds x = f(u[:, i], p[:, i], t)
    else
        @views @inbounds x = f(u[:, i], p[i], t)
    end
    @inbounds for j in 1:size(du, 1)
        du[j, i] = x[j]
    end
end

@kernel function jac_kernel(f, J, @Const(u),
    @Const(params::AbstractArray{ParamWrapper{P, T}}),
    @Const(t)) where {P, T}
    i = @index(Global, Linear) - 1
    section = (1 + (i * size(u, 1))):((i + 1) * size(u, 1))
    @inbounds p = params[i + 1].params
    @inbounds tspan = params[i + 1].data

    @views @inbounds f(J[section, section], u[:, i + 1], p, t)
    @inbounds for j in section, k in section
        J[k, j] = J[k, j] * (tspan[2] - tspan[1])
    end
end

@kernel function jac_kernel_oop(f, J, @Const(u),
    @Const(params::AbstractArray{ParamWrapper{P, T}}),
    @Const(t)) where {P, T}
    i = @index(Global, Linear) - 1
    section = (1 + (i * size(u, 1))):((i + 1) * size(u, 1))

    @inbounds p = params[i + 1].params
    @inbounds tspan = params[i + 1].data

    @views @inbounds x = f(u[:, i + 1], p, t)

    @inbounds for j in section, k in section
        J[k, j] = x[k, j] * (tspan[2] - tspan[1])
    end
end

@kernel function jac_kernel(f, J, @Const(u), @Const(p), @Const(t))
    i = @index(Global, Linear) - 1
    section = (1 + (i * size(u, 1))):((i + 1) * size(u, 1))
    if eltype(p) <: Number
        @views @inbounds f(J[section, section], u[:, i + 1], p[:, i + 1], t)
    else
        @views @inbounds f(J[section, section], u[:, i + 1], p[i + 1], t)
    end
end

@kernel function jac_kernel_oop(f, J, @Const(u), @Const(p), @Const(t))
    i = @index(Global, Linear) - 1
    section = (1 + (i * size(u, 1))):((i + 1) * size(u, 1))
    if eltype(p) <: Number
        @views @inbounds x = f(u[:, i + 1], p[:, i + 1], t)
    else
        @views @inbounds x = f(u[:, i + 1], p[i + 1], t)
    end
    @inbounds for j in section, k in section
        J[k, j] = x[k, j]
    end
end

@kernel function discrete_condition_kernel(condition, cur, @Const(u), @Const(t), @Const(p))
    i = @index(Global, Linear)
    @views @inbounds cur[i] = condition(u[:, i], t, FakeIntegrator(u[:, i], t, p[:, i]))
end

@kernel function discrete_affect!_kernel(affect!, cur, u, t, p)
    i = @index(Global, Linear)
    @views @inbounds cur[i] && affect!(FakeIntegrator(u[:, i], t, p[:, i]))
end

@kernel function continuous_condition_kernel(condition, out, @Const(u), @Const(t),
    @Const(p))
    i = @index(Global, Linear)
    @views @inbounds out[i] = condition(u[:, i], t, FakeIntegrator(u[:, i], t, p[:, i]))
end

@kernel function continuous_affect!_kernel(affect!, event_idx, u, t, p)
    for i in event_idx
        @views @inbounds affect!(FakeIntegrator(u[:, i], t, p[:, i]))
    end
end

maxthreads(::CPU) = 1024
maybe_prefer_blocks(::CPU) = CPU()

function workgroupsize(backend, n)
    min(maxthreads(backend), n)
end

@kernel function W_kernel(jac, W, @Const(u),
    @Const(params::AbstractArray{ParamWrapper{P, T}}), @Const(gamma),
    @Const(t)) where {P, T}
    i = @index(Global, Linear)
    len = size(u, 1)
    _W = @inbounds @view(W[:, :, i])

    @inbounds p = params[i].params
    @inbounds tspan = params[i].data

    @views @inbounds jac(_W, u[:, i], p, t)

    @inbounds for i in eachindex(_W)
        _W[i] = gamma * _W[i] * (tspan[2] - tspan[1])
    end
    _one = one(eltype(_W))
    @inbounds for i in 1:len
        _W[i, i] = _W[i, i] - _one
    end
end

@kernel function W_kernel(jac, W, @Const(u), @Const(p), @Const(gamma), @Const(t))
    i = @index(Global, Linear)
    len = size(u, 1)
    _W = @inbounds @view(W[:, :, i])
    @views @inbounds jac(_W, u[:, i], p[:, i], t)
    @inbounds for i in eachindex(_W)
        _W[i] = gamma * _W[i]
    end
    _one = one(eltype(_W))
    @inbounds for i in 1:len
        _W[i, i] = _W[i, i] - _one
    end
end

@kernel function W_kernel_oop(jac, W, @Const(u),
    @Const(params::AbstractArray{ParamWrapper{P, T}}),
    @Const(gamma),
    @Const(t)) where {P, T}
    i = @index(Global, Linear)
    len = size(u, 1)

    @inbounds p = params[i].params
    @inbounds tspan = params[i].data

    _W = @inbounds @view(W[:, :, i])

    @views @inbounds x = jac(u[:, i], p, t)
    @inbounds for j in 1:length(_W)
        _W[j] = x[j] * (tspan[2] - tspan[1])
    end
    @inbounds for i in eachindex(_W)
        _W[i] = gamma * _W[i]
    end
    _one = one(eltype(_W))
    @inbounds for i in 1:len
        _W[i, i] = _W[i, i] - _one
    end
end

@kernel function W_kernel_oop(jac, W, @Const(u), @Const(p), @Const(gamma), @Const(t))
    i = @index(Global, Linear)
    len = size(u, 1)
    _W = @inbounds @view(W[:, :, i])
    @views @inbounds x = jac(u[:, i], p[:, i], t)
    @inbounds for j in 1:length(_W)
        _W[j] = x[j]
    end
    @inbounds for i in eachindex(_W)
        _W[i] = gamma * _W[i]
    end
    _one = one(eltype(_W))
    @inbounds for i in 1:len
        _W[i, i] = _W[i, i] - _one
    end
end

@kernel function Wt_kernel(f::AbstractArray{T}, W, @Const(u), @Const(p), @Const(gamma),
    @Const(t)) where {T}
    i = @index(Global, Linear)
    len = size(u, 1)
    _W = @inbounds @view(W[:, :, i])
    @inbounds jac = f[i].tgrad
    @views @inbounds jac(_W, u[:, i], p[:, i], t)
    @inbounds for i in 1:len
        _W[i, i] = -inv(gamma) + _W[i, i]
    end
end

@kernel function Wt_kernel(jac, W, @Const(u), @Const(p), @Const(gamma), @Const(t))
    i = @index(Global, Linear)
    len = size(u, 1)
    _W = @inbounds @view(W[:, :, i])
    @views @inbounds jac(_W, u[:, i], p[:, i], t)
    @inbounds for i in 1:len
        _W[i, i] = -inv(gamma) + _W[i, i]
    end
end

@kernel function Wt_kernel_oop(f::AbstractArray{T}, W, @Const(u), @Const(p), @Const(gamma),
    @Const(t)) where {T}
    i = @index(Global, Linear)
    len = size(u, 1)
    _W = @inbounds @view(W[:, :, i])
    @inbounds jac = f[i].tgrad
    @views @inbounds x = jac(u[:, i], p[:, i], t)
    @inbounds for j in 1:length(_W)
        _W[j] = x[j]
    end
    @inbounds for i in 1:len
        _W[i, i] = -inv(gamma) + _W[i, i]
    end
end

@kernel function Wt_kernel_oop(jac, W, @Const(u), @Const(p), @Const(gamma), @Const(t))
    i = @index(Global, Linear)
    len = size(u, 1)
    _W = @inbounds @view(W[:, :, i])
    @views @inbounds x = jac(u[:, i], p[:, i], t)
    @inbounds for j in 1:length(_W)
        _W[j] = x[j]
    end
    @inbounds for i in 1:len
        _W[i, i] = -inv(gamma) + _W[i, i]
    end
end

@kernel function gpu_kernel_tgrad(f::AbstractArray{T}, du, @Const(u), @Const(p),
    @Const(t)) where {T}
    i = @index(Global, Linear)
    @inbounds f = f[i].tgrad
    if eltype(p) <: Number
        @views @inbounds f(du[:, i], u[:, i], p[:, i], t)
    else
        @views @inbounds f(du[:, i], u[:, i], p[i], t)
    end
end

@kernel function gpu_kernel_oop_tgrad(f::AbstractArray{T}, du, @Const(u), @Const(p),
    @Const(t)) where {T}
    i = @index(Global, Linear)
    @inbounds f = f[i].tgrad
    if eltype(p) <: Number
        @views @inbounds x = f(u[:, i], p[:, i], t)
    else
        @views @inbounds x = f(u[:, i], p[i], t)
    end
    @inbounds for j in 1:size(du, 1)
        du[j, i] = x[j]
    end
end

function lufact!(::CPU, W)
    len = size(W, 1)
    for i in 1:size(W, 3)
        _W = @inbounds @view(W[:, :, i])
        generic_lufact!(_W, len)
    end
    return nothing
end

struct FakeIntegrator{uType, tType, P}
    u::uType
    t::tType
    p::P
end

struct GPUDiscreteCallback{F1, F2, F3, F4, F5} <: SciMLBase.AbstractDiscreteCallback
    condition::F1
    affect!::F2
    initialize::F3
    finalize::F4
    save_positions::F5
    function GPUDiscreteCallback(condition::F1, affect!::F2,
        initialize::F3, finalize::F4,
        save_positions::F5) where {F1, F2, F3, F4, F5}
        if save_positions != (false, false)
            error("Callback `save_positions` are incompatible with kernel-based GPU ODE solvers due requiring static sizing. Please ensure `save_positions = (false,false)` is set in all callback definitions used with such solvers.")
        end
        new{F1, F2, F3, F4, F5}(condition,
            affect!, initialize, finalize, save_positions)
    end
end
function GPUDiscreteCallback(condition, affect!;
    initialize = SciMLBase.INITIALIZE_DEFAULT,
    finalize = SciMLBase.FINALIZE_DEFAULT,
    save_positions = (false, false))
    GPUDiscreteCallback(condition, affect!, initialize, finalize, save_positions)
end

function Base.convert(::Type{GPUDiscreteCallback}, x::T) where {T <: DiscreteCallback}
    GPUDiscreteCallback(x.condition, x.affect!, x.initialize, x.finalize,
        Tuple(x.save_positions))
end

struct GPUContinuousCallback{F1, F2, F3, F4, F5, F6, T, T2, T3, I, R} <:
       SciMLBase.AbstractContinuousCallback
    condition::F1
    affect!::F2
    affect_neg!::F3
    initialize::F4
    finalize::F5
    idxs::I
    rootfind::SciMLBase.RootfindOpt
    interp_points::Int
    save_positions::F6
    dtrelax::R
    abstol::T
    reltol::T2
    repeat_nudge::T3
    function GPUContinuousCallback(condition::F1, affect!::F2, affect_neg!::F3,
        initialize::F4, finalize::F5, idxs::I, rootfind,
        interp_points, save_positions::F6, dtrelax::R, abstol::T,
        reltol::T2,
        repeat_nudge::T3) where {F1, F2, F3, F4, F5, F6, T, T2,
        T3, I, R,
    }
        if save_positions != (false, false)
            error("Callback `save_positions` are incompatible with kernel-based GPU ODE solvers due requiring static sizing. Please ensure `save_positions = (false,false)` is set in all callback definitions used with such solvers.")
        end
        new{F1, F2, F3, F4, F5, F6, T, T2, T3, I, R}(condition,
            affect!, affect_neg!,
            initialize, finalize, idxs, rootfind,
            interp_points,
            save_positions,
            dtrelax, abstol, reltol, repeat_nudge)
    end
end

function GPUContinuousCallback(condition, affect!, affect_neg!;
    initialize = SciMLBase.INITIALIZE_DEFAULT,
    finalize = SciMLBase.FINALIZE_DEFAULT,
    idxs = nothing,
    rootfind = LeftRootFind,
    save_positions = (false, false),
    interp_points = 10,
    dtrelax = 1,
    abstol = 10eps(Float32), reltol = 0,
    repeat_nudge = 1 // 100)
    GPUContinuousCallback(condition, affect!, affect_neg!, initialize, finalize,
        idxs,
        rootfind, interp_points,
        save_positions,
        dtrelax, abstol, reltol, repeat_nudge)
end

function GPUContinuousCallback(condition, affect!;
    initialize = SciMLBase.INITIALIZE_DEFAULT,
    finalize = SciMLBase.FINALIZE_DEFAULT,
    idxs = nothing,
    rootfind = LeftRootFind,
    save_positions = (false, false),
    affect_neg! = affect!,
    interp_points = 10,
    dtrelax = 1,
    abstol = 10eps(Float32), reltol = 0, repeat_nudge = 1 // 100)
    GPUContinuousCallback(condition, affect!, affect_neg!, initialize, finalize, idxs,
        rootfind, interp_points,
        save_positions,
        dtrelax, abstol, reltol, repeat_nudge)
end

function Base.convert(::Type{GPUContinuousCallback}, x::T) where {T <: ContinuousCallback}
    GPUContinuousCallback(x.condition, x.affect!, x.affect_neg!, x.initialize, x.finalize,
        x.idxs, x.rootfind, x.interp_points,
        Tuple(x.save_positions), x.dtrelax, 100 * eps(Float32), x.reltol,
        x.repeat_nudge)
end

abstract type EnsembleArrayAlgorithm <: SciMLBase.EnsembleAlgorithm end
abstract type EnsembleKernelAlgorithm <: SciMLBase.EnsembleAlgorithm end

"""
```julia
EnsembleCPUArray(cpu_offload = 0.2)
```

An `EnsembleArrayAlgorithm` which utilizes the CPU kernels to parallelize each ODE solve
with their separate ODE integrator on each kernel. This method is meant to be a debugging
counterpart to `EnsembleGPUArray`, having the same behavior and using the same
KernelAbstractions.jl process to build the combined ODE, but without the restrictions of
`f` being a GPU-compatible kernel function.

It is unlikely that this method is useful beyond library development and debugging, as
almost any case should be faster with `EnsembleThreads` or `EnsembleDistributed`.
"""
struct EnsembleCPUArray <: EnsembleArrayAlgorithm end

"""
```julia
EnsembleGPUArray(backend,cpu_offload = 0.2)
```

An `EnsembleArrayAlgorithm` which utilizes the GPU kernels to parallelize each ODE solve
with their separate ODE integrator on each kernel.

## Positional Arguments

  - `backend`: the KernelAbstractions backend for performing the computation.
  - `cpu_offload`: the percentage of trajectories to offload to the CPU. Default is 0.2 or
    20% of trajectories.

## Limitations

`EnsembleGPUArray` requires being able to generate a kernel for `f` using
KernelAbstractons.jl and solving the resulting ODE defined over `CuArray` input types.
This introduces the following limitations on its usage:

  - Not all standard Julia `f` functions are allowed. Only Julia `f` functions which are
    capable of being compiled into a GPU kernel are allowed. This notably means that
    certain features of Julia can cause issues inside of kernel, like:

      + Allocating memory (building arrays)
      + Linear algebra (anything that calls BLAS)
      + Broadcast

  - Not all ODE solvers are allowed, only those from OrdinaryDiffEq.jl. The tested feature
    set from OrdinaryDiffEq.jl includes:

      + Explicit Runge-Kutta methods
      + Implicit Runge-Kutta methods
      + Rosenbrock methods
      + DiscreteCallbacks and ContinuousCallbacks
  - Stiff ODEs require the analytical solution of every derivative function it requires.
    For example, Rosenbrock methods require the Jacobian and the gradient with respect to
    time, and so these two functions are required to be given. Note that they can be
    generated by the
    [modelingtoolkitize](https://docs.juliadiffeq.org/latest/tutorials/advanced_ode_example/#Automatic-Derivation-of-Jacobian-Functions-1)
    approach.
  - To use multiple GPUs over clusters, one must manually set up one process per GPU. See the
    multi-GPU tutorial for more details.

!!! warn

    Callbacks with `terminate!` does not work well with `EnsembleGPUArray` as the entire
    integration will hault when any of the trajectories hault. Use with caution.

## Example

```julia
using DiffEqGPU, CUDA, OrdinaryDiffEq
function lorenz(du, u, p, t)
    du[1] = p[1] * (u[2] - u[1])
    du[2] = u[1] * (p[2] - u[3]) - u[2]
    du[3] = u[1] * u[2] - p[3] * u[3]
end

u0 = Float32[1.0;0.0;0.0]
tspan = (0.0f0,100.0f0)
p = [10.0f0,28.0f0,8/3f0]
prob = ODEProblem(lorenz,u0,tspan,p)
prob_func = (prob,i,repeat) -> remake(prob,p=rand(Float32,3).*p)
monteprob = EnsembleProblem(prob, prob_func = prob_func, safetycopy=false)
@time sol = solve(monteprob,Tsit5(),EnsembleGPUArray(CUDADevice()),trajectories=10_000,saveat=1.0f0)
```
"""
struct EnsembleGPUArray{Backend} <: EnsembleArrayAlgorithm
    backend::Backend
    cpu_offload::Float64
end

##Solvers for EnsembleGPUKernel
abstract type GPUODEAlgorithm <: DiffEqBase.AbstractODEAlgorithm end
abstract type GPUSDEAlgorithm <: DiffEqBase.AbstractSDEAlgorithm end
abstract type GPUODEImplicitAlgorithm{AD} <: GPUODEAlgorithm end

"""
GPUTsit5()

A specialized implementation of the 5th order `Tsit5` method specifically for kernel
generation with EnsembleGPUKernel. For a similar CPU implementation, see
SimpleATsit5 from SimpleDiffEq.jl.
"""
struct GPUTsit5 <: GPUODEAlgorithm end

"""
GPUVern7()

A specialized implementation of the 7th order `Vern7` method specifically for kernel
generation with EnsembleGPUKernel.
"""
struct GPUVern7 <: GPUODEAlgorithm end

"""
GPUVern9()

A specialized implementation of the 9th order `Vern9` method specifically for kernel
generation with EnsembleGPUKernel.
"""
struct GPUVern9 <: GPUODEAlgorithm end

"""
GPURosenbrock23()

A specialized implementation of the W-method `Rosenbrock23` method specifically for kernel
generation with EnsembleGPUKernel.
"""
struct GPURosenbrock23{AD} <: GPUODEImplicitAlgorithm{AD} end

"""
GPURodas4()

A specialized implementation of the `Rodas4` method specifically for kernel
generation with EnsembleGPUKernel.
"""
struct GPURodas4{AD} <: GPUODEImplicitAlgorithm{AD} end

"""
GPURodas5P()

A specialized implementation of the `Rodas5P` method specifically for kernel
generation with EnsembleGPUKernel.
"""
struct GPURodas5P{AD} <: GPUODEImplicitAlgorithm{AD} end

"""
GPUKvaerno3()

A specialized implementation of the `Kvaerno3` method specifically for kernel
generation with EnsembleGPUKernel.
"""
struct GPUKvaerno3{AD} <: GPUODEImplicitAlgorithm{AD} end

"""
GPUKvaerno5()

A specialized implementation of the `Kvaerno5` method specifically for kernel
generation with EnsembleGPUKernel.
"""
struct GPUKvaerno5{AD} <: GPUODEImplicitAlgorithm{AD} end

for Alg in [:GPURosenbrock23, :GPURodas4, :GPURodas5P, :GPUKvaerno3, :GPUKvaerno5]
    @eval begin
        function $Alg(; autodiff = Val{true}())
            $Alg{SciMLBase._unwrap_val(autodiff)}()
        end
    end
end

"""
GPUEM()

A specialized implementation of the Euler-Maruyama `GPUEM` method with weak order 1.0. Made specifically for kernel
generation with EnsembleGPUKernel.
"""
struct GPUEM <: GPUSDEAlgorithm end

"""
GPUSIEA()

A specialized implementation of the weak order 2.0 for Ito SDEs `GPUSIEA` method specifically for kernel
generation with EnsembleGPUKernel.
"""
struct GPUSIEA <: GPUSDEAlgorithm end

"""
```julia
EnsembleGPUKernel(backend,cpu_offload = 0.2)
```

A massively-parallel ensemble algorithm which generates a unique GPU kernel for the entire
ODE which is then executed. This leads to a very low overhead GPU code generation, but
imparts some extra limitations on the use.

## Positional Arguments

  - `backend`: the KernelAbstractions backend for performing the computation.
  - `cpu_offload`: the percentage of trajectories to offload to the CPU. Default is 0.2 or
    20% of trajectories.

## Limitations

  - Not all standard Julia `f` functions are allowed. Only Julia `f` functions which are
    capable of being compiled into a GPU kernel are allowed. This notably means that
    certain features of Julia can cause issues inside a kernel, like:

      + Allocating memory (building arrays)
      + Linear algebra (anything that calls BLAS)
      + Broadcast

  - Only out-of-place `f` definitions are allowed. Coupled with the requirement of not
    allowing for memory allocations, this means that the ODE must be defined with
    `StaticArray` initial conditions.
  - Only specific ODE solvers are allowed. This includes:

      + GPUTsit5
      + GPUVern7
      + GPUVern9
  - To use multiple GPUs over clusters, one must manually set up one process per GPU. See the
    multi-GPU tutorial for more details.

## Example

```julia
using DiffEqGPU, CUDA, OrdinaryDiffEq, StaticArrays

function lorenz(u, p, t)
    σ = p[1]
    ρ = p[2]
    β = p[3]
    du1 = σ * (u[2] - u[1])
    du2 = u[1] * (ρ - u[3]) - u[2]
    du3 = u[1] * u[2] - β * u[3]
    return SVector{3}(du1, du2, du3)
end

u0 = @SVector [1.0f0; 0.0f0; 0.0f0]
tspan = (0.0f0, 10.0f0)
p = @SVector [10.0f0, 28.0f0, 8 / 3.0f0]
prob = ODEProblem{false}(lorenz, u0, tspan, p)
prob_func = (prob, i, repeat) -> remake(prob, p = (@SVector rand(Float32, 3)) .* p)
monteprob = EnsembleProblem(prob, prob_func = prob_func, safetycopy = false)

@time sol = solve(monteprob, GPUTsit5(), EnsembleGPUKernel(), trajectories = 10_000,
                  adaptive = false, dt = 0.1f0)
```
"""
struct EnsembleGPUKernel{Dev} <: EnsembleKernelAlgorithm
    dev::Dev
    cpu_offload::Float64
end

cpu_alg = Dict(GPUTsit5 => (GPUSimpleTsit5(), GPUSimpleATsit5()),
    GPUVern7 => (GPUSimpleVern7(), GPUSimpleAVern7()),
    GPUVern9 => (GPUSimpleVern9(), GPUSimpleAVern9()))

# Work around the fact that Zygote cannot handle the task system
# Work around the fact that Zygote isderiving fails with constants?
function EnsembleGPUArray(dev)
    EnsembleGPUArray(dev, 0.2)
end

function EnsembleGPUKernel(dev)
    EnsembleGPUKernel(dev, 0.2)
end

function ChainRulesCore.rrule(::Type{<:EnsembleGPUArray})
    EnsembleGPUArray(0.0), _ -> NoTangent()
end

ZygoteRules.@adjoint function EnsembleGPUArray(dev)
    EnsembleGPUArray(dev, 0.0), _ -> nothing
end

function SciMLBase.__solve(ensembleprob::SciMLBase.AbstractEnsembleProblem,
    alg::Union{SciMLBase.DEAlgorithm, Nothing,
        DiffEqGPU.GPUODEAlgorithm, DiffEqGPU.GPUSDEAlgorithm},
    ensemblealg::Union{EnsembleArrayAlgorithm,
        EnsembleKernelAlgorithm};
    trajectories, batch_size = trajectories,
    unstable_check = (dt, u, p, t) -> false, adaptive = true,
    kwargs...)
    if trajectories == 1
        return SciMLBase.__solve(ensembleprob, alg, EnsembleSerial(); trajectories = 1,
            kwargs...)
    end

    cpu_trajectories = ((ensemblealg isa EnsembleGPUArray ||
                         ensemblealg isa EnsembleGPUKernel) &&
                        ensembleprob.reduction === SciMLBase.DEFAULT_REDUCTION) &&
                       (haskey(kwargs, :callback) ? kwargs[:callback] === nothing : true) ?
                       round(Int, trajectories * ensemblealg.cpu_offload) : 0
    gpu_trajectories = trajectories - cpu_trajectories

    num_batches = gpu_trajectories ÷ batch_size
    num_batches * batch_size != gpu_trajectories && (num_batches += 1)

    if cpu_trajectories != 0 && ensembleprob.reduction === SciMLBase.DEFAULT_REDUCTION
        cpu_II = (gpu_trajectories + 1):trajectories
        _alg = if typeof(alg) <: GPUODEAlgorithm
            if adaptive == false
                cpu_alg[typeof(alg)][1]
            else
                cpu_alg[typeof(alg)][2]
            end
        elseif typeof(alg) <: GPUSDEAlgorithm
            if adaptive == false
                SimpleEM()
            else
                error("Adaptive EM is not supported yet.")
            end
        else
            alg
        end

        function f()
            SciMLBase.solve_batch(ensembleprob, _alg, EnsembleThreads(), cpu_II, nothing;
                kwargs...)
        end

        cpu_sols = Channel{Core.Compiler.return_type(f, Tuple{})}(1)
        t = @task begin
            put!(cpu_sols, f())
        end
        schedule(t)
    end

    if num_batches == 1 && ensembleprob.reduction === SciMLBase.DEFAULT_REDUCTION
        time = @elapsed sol = batch_solve(ensembleprob, alg, ensemblealg,
            1:gpu_trajectories, adaptive;
            unstable_check = unstable_check, kwargs...)
        if cpu_trajectories != 0
            wait(t)
            sol = vcat(sol, take!(cpu_sols))
        end
        return SciMLBase.EnsembleSolution(sol, time, true)
    end

    converged::Bool = false
    u = ensembleprob.u_init === nothing ?
        similar(batch_solve(ensembleprob, alg, ensemblealg, 1:batch_size, adaptive;
            unstable_check = unstable_check, kwargs...), 0) :
        ensembleprob.u_init

    if nprocs() == 1
        # While pmap works, this makes much better error messages.
        time = @elapsed begin
            sols = map(1:num_batches) do i
                if i == num_batches
                    I = (batch_size * (i - 1) + 1):gpu_trajectories
                else
                    I = (batch_size * (i - 1) + 1):(batch_size * i)
                end
                batch_data = batch_solve(ensembleprob, alg, ensemblealg, I, adaptive;
                    unstable_check = unstable_check, kwargs...)
                if ensembleprob.reduction !== SciMLBase.DEFAULT_REDUCTION
                    u, _ = ensembleprob.reduction(u, batch_data, I)
                    return u
                else
                    batch_data
                end
            end
        end
    else
        time = @elapsed begin
            sols = pmap(1:num_batches) do i
                if i == num_batches
                    I = (batch_size * (i - 1) + 1):gpu_trajectories
                else
                    I = (batch_size * (i - 1) + 1):(batch_size * i)
                end
                x = batch_solve(ensembleprob, alg, ensemblealg, I, adaptive;
                    unstable_check = unstable_check, kwargs...)
                yield()
                if ensembleprob.reduction !== SciMLBase.DEFAULT_REDUCTION
                    u, _ = ensembleprob.reduction(u, x, I)
                else
                    x
                end
            end
        end
    end

    if ensembleprob.reduction === SciMLBase.DEFAULT_REDUCTION
        if cpu_trajectories != 0
            wait(t)
            sols = vcat(reduce(vcat, vec.(sols)), take!(cpu_sols))
        else
            sols = reduce(vcat, sols)
        end
        SciMLBase.EnsembleSolution(sols, time, true)
    else
        SciMLBase.EnsembleSolution(sols[end], time, true)
    end
end

diffeqgpunorm(u::AbstractArray, t) = sqrt.(sum(abs2, u) ./ length(u))
diffeqgpunorm(u::Union{AbstractFloat, Complex}, t) = abs(u)
function diffeqgpunorm(u::AbstractArray{<:ForwardDiff.Dual}, t)
    sqrt.(sum(abs2 ∘ ForwardDiff.value, u) ./ length(u))
end
diffeqgpunorm(u::ForwardDiff.Dual, t) = abs(ForwardDiff.value(u))

"""
Lower level API for `EnsembleArrayAlgorithm`. Avoids conversion of solution to CPU arrays.

```julia
vectorized_map_solve(probs, alg,
                     ensemblealg::Union{EnsembleArrayAlgorithm}, I,
                     adaptive)
```

## Arguments

- `probs`: the GPU-setup problems generated by the ensemble.
- `alg`: the kernel-based differential equation solver. Most of the solvers from OrdinaryDiffEq.jl
         are supported.
- `ensemblealg`: The `EnsembleGPUArray()` algorithm.
- `I`: The iterator argument. Can be set to for e.g. 1:10_000 to simulate 10,000 trajectories.
- `adaptive`: The Boolean argument for time-stepping. Use `true` to enable adaptive time-stepping.

## Keyword Arguments

Only a subset of the common solver arguments are supported.

"""
function vectorized_map_solve end

function vectorized_map_solve(probs, alg,
    ensemblealg::Union{EnsembleArrayAlgorithm}, I,
    adaptive;
    kwargs...)

    #    @assert all(Base.Fix2((prob1, prob2) -> isequal(prob1.tspan, prob2.tspan),probs[1]),probs)
    # u0 = reduce(hcat, Array(probs[i].u0) for i in 1:length(I))
    # p = reduce(hcat,
    #             probs[i].p isa SciMLBase.NullParameters ? probs[i].p : Array(probs[i].p)
    #             for i in 1:length(I))

    u0 = hcat([Array(probs[i].u0) for i in 1:length(I)]...)
    p = hcat([Array(probs[i].p) for i in 1:length(I)]...)

    prob = probs[1]
    sol = vectorized_map_solve_up(prob, alg, ensemblealg, I, u0, p;
        adaptive = adaptive, kwargs...)
end

function batch_solve(ensembleprob, alg,
    ensemblealg::Union{EnsembleArrayAlgorithm, EnsembleKernelAlgorithm}, I,
    adaptive;
    kwargs...)
    if ensembleprob.safetycopy
        probs = map(I) do i
            ensembleprob.prob_func(deepcopy(ensembleprob.prob), i, 1)
        end
    else
        probs = map(I) do i
            ensembleprob.prob_func(ensembleprob.prob, i, 1)
        end
    end
    @assert !isempty(I)
    #@assert all(p->p.f === probs[1].f,probs)

    if ensemblealg isa EnsembleGPUKernel
        # Using inner saveat requires all of them to be of same size,
        # because the dimension of CuMatrix is decided by it.
        # The columns of it are accessed at each thread.
        if !all(Base.Fix2((prob1, prob2) -> isequal(prob1.tspan, prob2.tspan),
                probs[1]),
            probs)
            if !iszero(ensemblealg.cpu_offload)
                error("Different time spans in an Ensemble Simulation with CPU offloading is not supported yet.")
            end
            if get(probs[1].kwargs, :saveat, nothing) === nothing && !adaptive &&
               get(kwargs, :save_everystep, true)
                error("Using different time-spans require either turning off save_everystep or using saveat. If using saveat, it should be of same length across the ensemble.")
            end
            if !all(Base.Fix2((prob1, prob2) -> isequal(sizeof(get(prob1.kwargs, :saveat,
                            nothing)),
                        sizeof(get(prob2.kwargs, :saveat,
                            nothing))), probs[1]),
                probs)
                error("Using different saveat in EnsembleGPUKernel requires all of them to be of same length. Use saveats of same size only.")
            end
        end

        if alg isa Union{GPUODEAlgorithm, GPUSDEAlgorithm}
            # Get inner saveat if global one isn't specified
            _saveat = get(probs[1].kwargs, :saveat, nothing)
            saveat = _saveat === nothing ? get(kwargs, :saveat, nothing) : _saveat
            solts, solus = batch_solve_up_kernel(ensembleprob, probs, alg, ensemblealg, I,
                adaptive; saveat = saveat, kwargs...)
            [begin
                ts = @view solts[:, i]
                us = @view solus[:, i]
                sol_idx = findlast(x -> x != probs[i].tspan[1], ts)
                if sol_idx === nothing
                    @error "No solution found" tspan=probs[i].tspan[1] ts
                    error("Batch solve failed")
                end
                @views ensembleprob.output_func(SciMLBase.build_solution(probs[i],
                        alg,
                        ts[1:sol_idx],
                        us[1:sol_idx],
                        k = nothing,
                        stats = nothing,
                        calculate_error = false,
                        retcode = sol_idx !=
                                  length(ts) ?
                                  ReturnCode.Terminated :
                                  ReturnCode.Success),
                    i)[1]
            end
             for i in eachindex(probs)]

        else
            error("We don't have solvers implemented for this algorithm yet")
        end
    else
        u0 = reduce(hcat, Array(probs[i].u0) for i in 1:length(I))

        if !all(Base.Fix2((prob1, prob2) -> isequal(prob1.tspan, prob2.tspan),
                probs[1]),
            probs)

            # Requires prob.p to be isbits otherwise it wouldn't work with ParamWrapper
            @assert all(prob -> isbits(prob.p), probs)
            @info "Remaking the problem to normalize time span values..."

            p = reduce(hcat,
                ParamWrapper(probs[i].p, probs[i].tspan)
                for i in 1:length(I))

            # Change the tspan of first problem to (0,1)
            orig_prob = copy(probs[1])
            probs[1] = remake(probs[1];
                tspan = (zero(probs[1].tspan[1]), one(probs[1].tspan[2])))

            sol, solus = batch_solve_up(ensembleprob, probs, alg, ensemblealg, I,
                u0, p; adaptive = adaptive, kwargs...)

            probs[1] = orig_prob

            [ensembleprob.output_func(SciMLBase.build_solution(probs[i], alg,
                    map(t -> probs[i].tspan[1] +
                             (probs[i].tspan[2] -
                              probs[i].tspan[1]) *
                             t, sol.t), solus[i],
                    stats = sol.stats,
                    retcode = sol.retcode), i)[1]
             for i in 1:length(probs)]
        else
            p = reduce(hcat,
                probs[i].p isa AbstractArray ? Array(probs[i].p) : probs[i].p
                for i in 1:length(I))
            sol, solus = batch_solve_up(ensembleprob, probs, alg, ensemblealg, I, u0, p;
                adaptive = adaptive, kwargs...)
            [ensembleprob.output_func(SciMLBase.build_solution(probs[i], alg, sol.t,
                    solus[i],
                    stats = sol.stats,
                    retcode = sol.retcode), i)[1]
             for i in 1:length(probs)]
        end
    end
end

function batch_solve_up_kernel(ensembleprob, probs, alg, ensemblealg, I, adaptive;
    kwargs...)
    _callback = CallbackSet(generate_callback(probs[1], length(I), ensemblealg; kwargs...))

    _callback = CallbackSet(convert.(DiffEqGPU.GPUDiscreteCallback,
            _callback.discrete_callbacks)...,
        convert.(DiffEqGPU.GPUContinuousCallback,
            _callback.continuous_callbacks)...)

    dev = ensemblealg.dev
    probs = adapt(dev, probs)

    #Adaptive version only works with saveat
    if adaptive
        ts, us = vectorized_asolve(probs, ensembleprob.prob, alg;
            kwargs..., callback = _callback)
    else
        ts, us = vectorized_solve(probs, ensembleprob.prob, alg;
            kwargs..., callback = _callback)
    end
    solus = Array(us)
    solts = Array(ts)
    (solts, solus)
end

function vectorized_map_solve_up(prob, alg, ensemblealg, I, u0, p; kwargs...)
    if ensemblealg isa EnsembleGPUArray
        backend = ensemblealg.backend
        u0 = adapt(backend, u0)
        p = adapt(backend, p)
    end

    len = length(prob.u0)

    if SciMLBase.has_jac(prob.f)
        if ensemblealg isa EnsembleGPUArray
            backend = ensemblealg.backend
            jac_prototype = allocate(backend, Float32, (len, len, length(I)))
            fill!(jac_prototype, 0.0)
        else
            jac_prototype = zeros(Float32, len, len, length(I))
        end

        if prob.f.colorvec !== nothing
            colorvec = repeat(prob.f.colorvec, length(I))
        else
            colorvec = repeat(1:length(prob.u0), length(I))
        end
    else
        jac_prototype = nothing
        colorvec = nothing
    end

    _callback = generate_callback(prob, length(I), ensemblealg; kwargs...)
    prob = generate_problem(prob, u0, p, jac_prototype, colorvec)

    if hasproperty(alg, :linsolve)
        _alg = remake(alg, linsolve = LinSolveGPUSplitFactorize(len, -1))
    else
        _alg = alg
    end

    sol = solve(prob, _alg; kwargs..., callback = _callback, merge_callbacks = false,
        internalnorm = diffeqgpunorm)
end

function batch_solve_up(ensembleprob, probs, alg, ensemblealg, I, u0, p; kwargs...)
    if ensemblealg isa EnsembleGPUArray
        backend = ensemblealg.backend
        u0 = adapt(backend, u0)
        p = adapt(backend, p)
    end

    len = length(probs[1].u0)

    if SciMLBase.has_jac(probs[1].f)
        if ensemblealg isa EnsembleGPUArray
            backend = ensemblealg.backend
            jac_prototype = allocate(backend, Float32, (len, len, length(I)))
            fill!(jac_prototype, 0.0)
        else
            jac_prototype = zeros(Float32, len, len, length(I))
        end

        if probs[1].f.colorvec !== nothing
            colorvec = repeat(probs[1].f.colorvec, length(I))
        else
            colorvec = repeat(1:length(probs[1].u0), length(I))
        end
    else
        jac_prototype = nothing
        colorvec = nothing
    end

    _callback = generate_callback(probs[1], length(I), ensemblealg; kwargs...)
    prob = generate_problem(probs[1], u0, p, jac_prototype, colorvec)

    if hasproperty(alg, :linsolve)
        _alg = remake(alg, linsolve = LinSolveGPUSplitFactorize(len, -1))
    else
        _alg = alg
    end

    sol = solve(prob, _alg; kwargs..., callback = _callback, merge_callbacks = false,
        internalnorm = diffeqgpunorm)

    us = Array.(sol.u)
    solus = [[@view(us[i][:, j]) for i in 1:length(us)] for j in 1:length(probs)]
    (sol, solus)
end

function seed_duals(x::Matrix{V}, ::Type{T},
    ::ForwardDiff.Chunk{N} = ForwardDiff.Chunk(@view(x[:, 1]),
        typemax(Int64))) where {V, T,
    N}
    seeds = ForwardDiff.construct_seeds(ForwardDiff.Partials{N, V})
    duals = [ForwardDiff.Dual{T}(x[i, j], seeds[i])
             for i in 1:size(x, 1), j in 1:size(x, 2)]
end

function extract_dus(us)
    jsize = size(us[1], 1), ForwardDiff.npartials(us[1][1])
    utype = typeof(ForwardDiff.value(us[1][1]))
    map(1:size(us[1], 2)) do k
        map(us) do u
            du_i = zeros(utype, jsize)
            for i in size(u, 1)
                du_i[i, :] = ForwardDiff.partials(u[i, k])
            end
            du_i
        end
    end
end

struct DiffEqGPUAdjTag end

function ChainRulesCore.rrule(::typeof(batch_solve_up), ensembleprob, probs, alg,
    ensemblealg, I, u0, p; kwargs...)
    pdual = seed_duals(p, DiffEqGPUAdjTag)
    u0 = convert.(eltype(pdual), u0)

    if ensemblealg isa EnsembleGPUArray
        backend = ensemblealg.backend
        u0 = adapt(backend, u0)
        pdual = adapt(backend, pdual)
    end

    len = length(probs[1].u0)

    if SciMLBase.has_jac(probs[1].f)
        if ensemblealg isa EnsembleGPUArray
            backend = ensemblealg.backend
            jac_prototype = allocate(backend, Float32, (len, len, length(I)))
            fill!(jac_prototype, 0.0)
        else
            jac_prototype = zeros(Float32, len, len, length(I))
        end
        if probs[1].f.colorvec !== nothing
            colorvec = repeat(probs[1].f.colorvec, length(I))
        else
            colorvec = repeat(1:length(probs[1].u0), length(I))
        end
    else
        jac_prototype = nothing
        colorvec = nothing
    end

    _callback = generate_callback(probs[1], length(I), ensemblealg)
    prob = generate_problem(probs[1], u0, pdual, jac_prototype, colorvec)

    if hasproperty(alg, :linsolve)
        _alg = remake(alg, linsolve = LinSolveGPUSplitFactorize(len, -1))
    else
        _alg = alg
    end

    sol = solve(prob, _alg; kwargs..., callback = _callback, merge_callbacks = false,
        internalnorm = diffeqgpunorm)

    us = Array.(sol.u)
    solus = [[ForwardDiff.value.(@view(us[i][:, j])) for i in 1:length(us)]
             for j in 1:length(probs)]

    function batch_solve_up_adjoint(Δ)
        dus = extract_dus(us)
        _Δ = Δ[2]
        adj = map(eachindex(dus)) do j
            sum(eachindex(dus[j])) do i
                J = dus[j][i]
                if _Δ[j] isa AbstractVector
                    v = _Δ[j][i]
                else
                    v = @view _Δ[j][i]
                end
                J'v
            end
        end
        (ntuple(_ -> NoTangent(), 7)..., Array(VectorOfArray(adj)))
    end
    (sol, solus), batch_solve_up_adjoint
end

function generate_problem(prob::ODEProblem, u0, p, jac_prototype, colorvec)
    _f = let f = prob.f.f, kernel = DiffEqBase.isinplace(prob) ? gpu_kernel : gpu_kernel_oop
        function (du, u, p, t)
            version = get_backend(u)
            wgs = workgroupsize(version, size(u, 2))
            kernel(version)(f, du, u, p, t; ndrange = size(u, 2),
                workgroupsize = wgs)
        end
    end

    if SciMLBase.has_jac(prob.f)
        _Wfact! = let jac = prob.f.jac,
            kernel = DiffEqBase.isinplace(prob) ? W_kernel : W_kernel_oop

            function (W, u, p, gamma, t)
                version = get_backend(u)
                wgs = workgroupsize(version, size(u, 2))
                kernel(version)(jac, W, u, p, gamma, t;
                    ndrange = size(u, 2),
                    workgroupsize = wgs)
                lufact!(version, W)
            end
        end
        _Wfact!_t = let jac = prob.f.jac,
            kernel = DiffEqBase.isinplace(prob) ? Wt_kernel : Wt_kernel_oop

            function (W, u, p, gamma, t)
                version = get_backend(u)
                wgs = workgroupsize(version, size(u, 2))
                kernel(version)(jac, W, u, p, gamma, t;
                    ndrange = size(u, 2),
                    workgroupsize = wgs)
                lufact!(version, W)
            end
        end
    else
        _Wfact! = nothing
        _Wfact!_t = nothing
    end

    if SciMLBase.has_tgrad(prob.f)
        _tgrad = let tgrad = prob.f.tgrad,
            kernel = DiffEqBase.isinplace(prob) ? gpu_kernel : gpu_kernel_oop

            function (J, u, p, t)
                version = get_backend(u)
                wgs = workgroupsize(version, size(u, 2))
                kernel(version)(tgrad, J, u, p, t;
                    ndrange = size(u, 2),
                    workgroupsize = wgs)
            end
        end
    else
        _tgrad = nothing
    end

    f_func = ODEFunction(_f, Wfact = _Wfact!,
        Wfact_t = _Wfact!_t,
        #colorvec=colorvec,
        jac_prototype = jac_prototype,
        tgrad = _tgrad)
    prob = ODEProblem(f_func, u0, prob.tspan, p;
        prob.kwargs...)
end

function generate_problem(prob::SDEProblem, u0, p, jac_prototype, colorvec)
    _f = let f = prob.f.f, kernel = DiffEqBase.isinplace(prob) ? gpu_kernel : gpu_kernel_oop
        function (du, u, p, t)
            version = get_backend(u)
            wgs = workgroupsize(version, size(u, 2))
            kernel(version)(f, du, u, p, t;
                ndrange = size(u, 2),
                workgroupsize = wgs)
        end
    end

    _g = let f = prob.f.g, kernel = DiffEqBase.isinplace(prob) ? gpu_kernel : gpu_kernel_oop
        function (du, u, p, t)
            version = get_backend(u)
            wgs = workgroupsize(version, size(u, 2))
            kernel(version)(f, du, u, p, t;
                ndrange = size(u, 2),
                workgroupsize = wgs)
        end
    end

    if SciMLBase.has_jac(prob.f)
        _Wfact! = let jac = prob.f.jac,
            kernel = DiffEqBase.isinplace(prob) ? W_kernel : W_kernel_oop

            function (W, u, p, gamma, t)
                version = get_backend(u)
                wgs = workgroupsize(version, size(u, 2))
                kernel(version)(jac, W, u, p, gamma, t;
                    ndrange = size(u, 2),
                    workgroupsize = wgs)
                lufact!(version, W)
            end
        end
        _Wfact!_t = let jac = prob.f.jac,
            kernel = DiffEqBase.isinplace(prob) ? Wt_kernel : Wt_kernel_oop

            function (W, u, p, gamma, t)
                version = get_backend(u)
                wgs = workgroupsize(version, size(u, 2))
                kernel(version)(jac, W, u, p, gamma, t;
                    ndrange = size(u, 2),
                    workgroupsize = wgs)
                lufact!(version, W)
            end
        end
    else
        _Wfact! = nothing
        _Wfact!_t = nothing
    end

    if SciMLBase.has_tgrad(prob.f)
        _tgrad = let tgrad = prob.f.tgrad,
            kernel = DiffEqBase.isinplace(prob) ? gpu_kernel : gpu_kernel_oop

            function (J, u, p, t)
                version = get_backend(u)
                wgs = workgroupsize(version, size(u, 2))
                kernel(version)(tgrad, J, u, p, t;
                    ndrange = size(u, 2),
                    workgroupsize = wgs)
            end
        end
    else
        _tgrad = nothing
    end

    f_func = SDEFunction(_f, _g, Wfact = _Wfact!,
        Wfact_t = _Wfact!_t,
        #colorvec=colorvec,
        jac_prototype = jac_prototype,
        tgrad = _tgrad)
    prob = SDEProblem(f_func, _g, u0, prob.tspan, p;
        prob.kwargs...)
end

function generate_callback(callback::DiscreteCallback, I,
    ensemblealg)
    if ensemblealg isa EnsembleGPUArray
        backend = ensemblealg.backend
        cur = adapt(backend, [false for i in 1:I])
    elseif ensemblealg isa EnsembleGPUKernel
        return callback
    else
        cur = [false for i in 1:I]
    end
    _condition = callback.condition
    _affect! = callback.affect!

    condition = function (u, t, integrator)
        version = get_backend(u)
        wgs = workgroupsize(version, size(u, 2))
        discrete_condition_kernel(version)(_condition, cur, u, t, integrator.p;
            ndrange = size(u, 2),
            workgroupsize = wgs)
        any(cur)
    end

    affect! = function (integrator)
        version = get_backend(integrator.u)
        wgs = workgroupsize(version, size(integrator.u, 2))
        discrete_affect!_kernel(version)(_affect!, cur, integrator.u, integrator.t,
            integrator.p;
            ndrange = size(integrator.u, 2),
            workgroupsize = wgs)
    end
    return DiscreteCallback(condition, affect!, save_positions = callback.save_positions)
end

function generate_callback(callback::ContinuousCallback, I, ensemblealg)
    if ensemblealg isa EnsembleGPUKernel
        return callback
    end
    _condition = callback.condition
    _affect! = callback.affect!
    _affect_neg! = callback.affect_neg!

    condition = function (out, u, t, integrator)
        version = get_backend(u)
        wgs = workgroupsize(version, size(u, 2))
        continuous_condition_kernel(version)(_condition, out, u, t, integrator.p;
            ndrange = size(u, 2),
            workgroupsize = wgs)
        nothing
    end

    affect! = function (integrator, event_idx)
        version = get_backend(integrator.u)
        wgs = workgroupsize(version, size(integrator.u, 2))
        continuous_affect!_kernel(version)(_affect!, event_idx, integrator.u,
            integrator.t, integrator.p;
            ndrange = size(integrator.u, 2),
            workgroupsize = wgs)
    end

    affect_neg! = function (integrator, event_idx)
        version = get_backend(integrator.u)
        wgs = workgroupsize(version, size(integrator.u, 2))
        continuous_affect!_kernel(version)(_affect_neg!, event_idx, integrator.u,
            integrator.t, integrator.p;
            ndrange = size(integrator.u, 2),
            workgroupsize = wgs)
    end

    return VectorContinuousCallback(condition, affect!, affect_neg!, I,
        save_positions = callback.save_positions)
end

function generate_callback(callback::CallbackSet, I, ensemblealg)
    return CallbackSet(map(cb -> generate_callback(cb, I, ensemblealg),
        (callback.continuous_callbacks...,
            callback.discrete_callbacks...))...)
end

generate_callback(::Tuple{}, I, ensemblealg) = nothing

function generate_callback(x)
    # will catch any VectorContinuousCallbacks
    error("Callback unsupported")
end

function generate_callback(prob, I, ensemblealg; kwargs...)
    prob_cb = get(prob.kwargs, :callback, ())
    kwarg_cb = get(kwargs, :merge_callbacks, false) ? get(kwargs, :callback, ()) : ()

    if (prob_cb === nothing || isempty(prob_cb)) &&
       (kwarg_cb === nothing || isempty(kwarg_cb))
        return nothing
    else
        return CallbackSet(generate_callback(prob_cb, I, ensemblealg),
            generate_callback(kwarg_cb, I, ensemblealg))
    end
end

### GPU Factorization
"""
A parameter-parallel `SciMLLinearSolveAlgorithm`.
"""
struct LinSolveGPUSplitFactorize <: LinearSolve.SciMLLinearSolveAlgorithm
    len::Int
    nfacts::Int
end
LinSolveGPUSplitFactorize() = LinSolveGPUSplitFactorize(0, 0)

LinearSolve.needs_concrete_A(::LinSolveGPUSplitFactorize) = true

function LinearSolve.init_cacheval(linsol::LinSolveGPUSplitFactorize, A, b, u, Pl, Pr,
    maxiters::Int, abstol, reltol, verbose::Bool,
    assumptions::LinearSolve.OperatorAssumptions)
    LinSolveGPUSplitFactorize(linsol.len, length(u) ÷ linsol.len)
end

function SciMLBase.solve!(cache::LinearSolve.LinearCache, alg::LinSolveGPUSplitFactorize,
    args...; kwargs...)
    p = cache.cacheval
    A = cache.A
    b = cache.b
    x = cache.u
    version = get_backend(b)
    copyto!(x, b)
    wgs = workgroupsize(version, p.nfacts)
    # Note that the matrix is already factorized, only ldiv is needed.
    ldiv!_kernel(version)(A, x, p.len, p.nfacts;
        ndrange = p.nfacts,
        workgroupsize = wgs)
    SciMLBase.build_linear_solution(alg, x, nothing, cache)
end

# Old stuff
function (p::LinSolveGPUSplitFactorize)(x, A, b, update_matrix = false; kwargs...)
    version = get_backend(b)
    copyto!(x, b)
    wgs = workgroupsize(version, p.nfacts)
    ldiv!_kernel(version)(A, x, p.len, p.nfacts;
        ndrange = p.nfacts,
        workgroupsize = wgs)
    return nothing
end

function (p::LinSolveGPUSplitFactorize)(::Type{Val{:init}}, f, u0_prototype)
    LinSolveGPUSplitFactorize(size(u0_prototype)...)
end

@kernel function ldiv!_kernel(W, u, @Const(len), @Const(nfacts))
    i = @index(Global, Linear)
    section = (1 + ((i - 1) * len)):(i * len)
    _W = @inbounds @view(W[:, :, i])
    _u = @inbounds @view u[section]
    naivesolve!(_W, _u, len)
end

function generic_lufact!(A::AbstractMatrix{T}, minmn) where {T}
    m = n = minmn
    #@cuprintf "\n\nbefore lufact!\n"
    #__printjac(A, ii)
    #@cuprintf "\n"
    @inbounds for k in 1:minmn
        #@cuprintf "inner factorization loop\n"
        # Scale first column
        Akkinv = inv(A[k, k])
        for i in (k + 1):m
            #@cuprintf "L\n"
            A[i, k] *= Akkinv
        end
        # Update the rest
        for j in (k + 1):n, i in (k + 1):m
            #@cuprintf "U\n"
            A[i, j] -= A[i, k] * A[k, j]
        end
    end
    #@cuprintf "after lufact!"
    #__printjac(A, ii)
    #@cuprintf "\n\n\n"
    return nothing
end

struct MyL{T} # UnitLowerTriangular
    data::T
end
struct MyU{T} # UpperTriangular
    data::T
end

function naivesub!(A::MyU, b::AbstractVector, n)
    x = b
    @inbounds for j in n:-1:1
        xj = x[j] = A.data[j, j] \ b[j]
        for i in (j - 1):-1:1 # counterintuitively 1:j-1 performs slightly better
            b[i] -= A.data[i, j] * xj
        end
    end
    return nothing
end
function naivesub!(A::MyL, b::AbstractVector, n)
    x = b
    @inbounds for j in 1:n
        xj = x[j]
        for i in (j + 1):n
            b[i] -= A.data[i, j] * xj
        end
    end
    return nothing
end

function naivesolve!(A::AbstractMatrix, x::AbstractVector, n)
    naivesub!(MyL(A), x, n)
    naivesub!(MyU(A), x, n)
    return nothing
end

function solve_batch(prob, alg, ensemblealg::EnsembleThreads, II, pmap_batch_size;
    kwargs...)
    if length(II) == 1 || Threads.nthreads() == 1
        return SciMLBase.solve_batch(prob, alg, EnsembleSerial(), II, pmap_batch_size;
            kwargs...)
    end

    if typeof(prob.prob) <: SciMLBase.AbstractJumpProblem && length(II) != 1
        probs = [deepcopy(prob.prob) for i in 1:Threads.nthreads()]
    else
        probs = prob.prob
    end

    #
    batch_size = length(II) ÷ (Threads.nthreads() - 1)

    batch_data = tmap(1:(Threads.nthreads() - 1)) do i
        if i == Threads.nthreads() - 1
            I_local = II[(batch_size * (i - 1) + 1):end]
        else
            I_local = II[(batch_size * (i - 1) + 1):(batch_size * i)]
        end
        SciMLBase.solve_batch(prob, alg, EnsembleSerial(), I_local, pmap_batch_size;
            kwargs...)
    end
    SciMLBase.tighten_container_eltype(batch_data)
end

function tmap(f, args...)
    batch_data = Vector{Core.Compiler.return_type(f, Tuple{typeof.(getindex.(args, 1))...})
    }(undef,
        length(args[1]))
    Threads.@threads for i in 1:length(args[1])
        batch_data[i] = f(getindex.(args, i)...)
    end
    reduce(vcat, batch_data)
end

include("alg_utils.jl")
include("integrators/types.jl")
include("integrators/stiff/types.jl")
include("integrators/integrator_utils.jl")
include("integrators/interpolants.jl")
include("nlsolve/type.jl")
include("nlsolve/utils.jl")

include("perform_step/gpu_tsit5_perform_step.jl")
include("perform_step/gpu_vern7_perform_step.jl")
include("perform_step/gpu_vern9_perform_step.jl")
include("perform_step/gpu_em_perform_step.jl")
include("perform_step/gpu_siea_perform_step.jl")
include("perform_step/gpu_rosenbrock23_perform_step.jl")
include("perform_step/gpu_rodas4_perform_step.jl")
include("perform_step/gpu_rodas5P_perform_step.jl")
include("perform_step/gpu_kvaerno3_perform_step.jl")
include("perform_step/gpu_kvaerno5_perform_step.jl")

include("tableaus/verner_tableaus.jl")
include("tableaus/rodas_tableaus.jl")
include("tableaus/kvaerno_tableaus.jl")
include("solve.jl")

export EnsembleCPUArray, EnsembleGPUArray, EnsembleGPUKernel, LinSolveGPUSplitFactorize

export GPUTsit5, GPUVern7, GPUVern9, GPUEM, GPUSIEA
## Stiff ODE solvers
export GPURosenbrock23, GPURodas4, GPURodas5P, GPUKvaerno3, GPUKvaerno5

export terminate!

# This symbol is only defined on Julia versions that support extensions
if !isdefined(Base, :get_extension)
    using Requires
end

@static if !isdefined(Base, :get_extension)
    function __init__()
        @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" include("../ext/CUDAExt.jl")
        @require AMDGPU="21141c5a-9bdb-4563-92ae-f87d6854732e" include("../ext/AMDGPUExt.jl")
        @require oneAPI="8f75cd03-7ff8-4ecb-9b8f-daf728133b1b" include("../ext/oneAPIExt.jl")
        @require Metal="dde4c033-4e86-420c-a63e-0dd931031962" include("../ext/MetalExt.jl")
    end
end

end # module

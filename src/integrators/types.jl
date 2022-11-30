## Fixed TimeStep Integrator

function Adapt.adapt_structure(to, prob::ODEProblem{<:Any, <:Any, iip}) where {iip}
    ODEProblem{iip, true}(adapt(to, prob.f),
                          adapt(to, prob.u0),
                          adapt(to, prob.tspan),
                          adapt(to, prob.p);
                          adapt(to, prob.kwargs)...)
end

mutable struct GPUTsit5Integrator{IIP, S, T, ST, P, F, TS, CB} <:
               DiffEqBase.AbstractODEIntegrator{GPUTsit5, IIP, S, T}
    f::F                  # eom
    uprev::S              # previous state
    u::S                  # current state
    tmp::S                # dummy, same as state
    tprev::T              # previous time
    t::T                  # current time
    t0::T                 # initial time, only for reinit
    dt::T                 # step size
    tdir::T
    p::P                  # parameter container
    u_modified::Bool
    tstops::TS
    tstops_idx::Int
    callback::CB
    save_everystep::Bool
    saveat::ST
    cur_t::Int
    step_idx::Int
    event_last_time::Int
    vector_event_last_time::Int
    last_event_error::T
    k1::S                 #intepolants
    k2::S
    k3::S
    k4::S
    k5::S
    k6::S
    k7::S
    cs::SVector{6, T}     # ci factors cache: time coefficients
    as::SVector{21, T}    # aij factors cache: solution coefficients
    rs::SVector{22, T}    # rij factors cache: interpolation coefficients
end
const GPUT5I = GPUTsit5Integrator

function (integrator::GPUTsit5Integrator)(t)
    Θ = (t - integrator.tprev) / integrator.dt
    _ode_interpolant(Θ, integrator.dt, integrator.uprev, integrator)
end

function DiffEqBase.u_modified!(integrator::GPUTsit5Integrator, bool::Bool)
    integrator.u_modified = bool
end

DiffEqBase.isinplace(::GPUT5I{IIP}) where {IIP} = IIP

## Adaptive TimeStep Integrator

mutable struct GPUATsit5Integrator{IIP, S, T, ST, P, F, N, TOL, Q, TS, CB} <:
               DiffEqBase.AbstractODEIntegrator{GPUTsit5, IIP, S, T}
    f::F                  # eom
    uprev::S              # previous state
    u::S                  # current state
    tmp::S                # dummy, same as state
    tprev::T              # previous time
    t::T                  # current time
    t0::T                 # initial time, only for reinit
    tf::T
    dt::T                 # step size
    dtnew::T
    tdir::T
    p::P                  # parameter container
    u_modified::Bool
    tstops::TS
    tstops_idx::Int
    callback::CB
    save_everystep::Bool
    saveat::ST
    cur_t::Int
    step_idx::Int
    event_last_time::Int
    vector_event_last_time::Int
    last_event_error::T
    k1::S         # interpolants of the algorithm
    k2::S
    k3::S
    k4::S
    k5::S
    k6::S
    k7::S
    cs::SVector{6, T}     # ci factors cache: time coefficients
    as::SVector{21, T}    # aij factors cache: solution coefficients
    btildes::SVector{7, T}
    rs::SVector{22, T}    # rij factors cache: interpolation coefficients
    qold::Q
    abstol::TOL
    reltol::TOL
    internalnorm::N       # function that computes the error EEst based on state
end

const GPUAT5I = GPUATsit5Integrator

function (integrator::GPUATsit5Integrator)(t)
    Θ = (t - integrator.tprev) / integrator.dt
    _ode_interpolant(Θ, integrator.dt, integrator.uprev, integrator)
end

function DiffEqBase.u_modified!(integrator::GPUATsit5Integrator, bool::Bool)
    integrator.u_modified = bool
end
## Vern7

mutable struct GPUV7Integrator{IIP, S, T, ST, P, F, TS, CB, TabType} <:
               DiffEqBase.AbstractODEIntegrator{GPUVern7, IIP, S, T}
    f::F                  # eom
    uprev::S              # previous state
    u::S                  # current state
    tmp::S                # dummy, same as state
    tprev::T              # previous time
    t::T                  # current time
    t0::T                 # initial time, only for reinit
    dt::T                 # step size
    tdir::T
    p::P                  # parameter container
    u_modified::Bool
    tstops::TS
    tstops_idx::Int
    callback::CB
    save_everystep::Bool
    saveat::ST
    cur_t::Int
    step_idx::Int
    event_last_time::Int
    vector_event_last_time::Int
    last_event_error::T
    k1::S                 #intepolants
    k2::S
    k3::S
    k4::S
    k5::S
    k6::S
    k7::S
    k8::S
    k9::S
    k10::S
    tab::TabType
end
const GPUV7I = GPUV7Integrator

function (integrator::GPUV7I)(t)
    Θ = (t - integrator.tprev) / integrator.dt
    _ode_interpolant(Θ, integrator.dt, integrator.uprev, integrator)
end

mutable struct GPUAV7Integrator{IIP, S, T, ST, P, F, N, TOL, Q, TS, CB, TabType} <:
               DiffEqBase.AbstractODEIntegrator{GPUVern7, IIP, S, T}
    f::F                  # eom
    uprev::S              # previous state
    u::S                  # current state
    tmp::S                # dummy, same as state
    tprev::T              # previous time
    t::T                  # current time
    t0::T                 # initial time, only for reinit
    tf::T
    dt::T                 # step size
    dtnew::T
    tdir::T
    p::P                  # parameter container
    u_modified::Bool
    tstops::TS
    tstops_idx::Int
    callback::CB
    save_everystep::Bool
    saveat::ST
    cur_t::Int
    step_idx::Int
    event_last_time::Int
    vector_event_last_time::Int
    last_event_error::T
    k1::S         # interpolants of the algorithm
    k2::S
    k3::S
    k4::S
    k5::S
    k6::S
    k7::S
    k8::S
    k9::S
    k10::S
    tab::TabType
    qold::Q
    abstol::TOL
    reltol::TOL
    internalnorm::N       # function that computes the error EEst based on state
end

const GPUAV7I = GPUAV7Integrator

function (integrator::GPUAV7I)(t)
    Θ = (t - integrator.tprev) / integrator.dt
    _ode_interpolant(Θ, integrator.dt, integrator.uprev, integrator)
end

## Vern9

mutable struct GPUV9Integrator{IIP, S, T, ST, P, F, TS, CB, TabType} <:
               DiffEqBase.AbstractODEIntegrator{GPUVern9, IIP, S, T}
    f::F                  # eom
    uprev::S              # previous state
    u::S                  # current state
    tmp::S                # dummy, same as state
    tprev::T              # previous time
    t::T                  # current time
    t0::T                 # initial time, only for reinit
    dt::T                 # step size
    tdir::T
    p::P                  # parameter container
    u_modified::Bool
    tstops::TS
    tstops_idx::Int
    callback::CB
    save_everystep::Bool
    saveat::ST
    cur_t::Int
    step_idx::Int
    event_last_time::Int
    vector_event_last_time::Int
    last_event_error::T
    k1::S                 #intepolants
    k2::S
    k3::S
    k4::S
    k5::S
    k6::S
    k7::S
    k8::S
    k9::S
    k10::S
    tab::TabType
end
const GPUV9I = GPUV9Integrator

function (integrator::GPUV9I)(t)
    Θ = (t - integrator.tprev) / integrator.dt
    _ode_interpolant(Θ, integrator.dt, integrator.uprev, integrator)
end

mutable struct GPUAV9Integrator{IIP, S, T, ST, P, F, N, TOL, Q, TS, CB, TabType} <:
               DiffEqBase.AbstractODEIntegrator{GPUVern9, IIP, S, T}
    f::F                  # eom
    uprev::S              # previous state
    u::S                  # current state
    tmp::S                # dummy, same as state
    tprev::T              # previous time
    t::T                  # current time
    t0::T                 # initial time, only for reinit
    tf::T
    dt::T                 # step size
    dtnew::T
    tdir::T
    p::P                  # parameter container
    u_modified::Bool
    tstops::TS
    tstops_idx::Int
    callback::CB
    save_everystep::Bool
    saveat::ST
    cur_t::Int
    step_idx::Int
    event_last_time::Int
    vector_event_last_time::Int
    last_event_error::T
    k1::S         # interpolants of the algorithm
    k2::S
    k3::S
    k4::S
    k5::S
    k6::S
    k7::S
    k8::S
    k9::S
    k10::S
    tab::TabType
    qold::Q
    abstol::TOL
    reltol::TOL
    internalnorm::N       # function that computes the error EEst based on state
end

const GPUAV9I = GPUAV9Integrator

function (integrator::GPUAV9I)(t)
    Θ = (t - integrator.tprev) / integrator.dt
    _ode_interpolant(Θ, integrator.dt, integrator.uprev, integrator)
end

#######################################################################################
# Initialization of Integrators
#######################################################################################
@inline function gputsit5_init(f::F, IIP::Bool, u0::S, t0::T, dt::T,
                               p::P, tstops::TS,
                               callback::CB,
                               save_everystep::Bool,
                               saveat::ST) where {F, P, T, S <: AbstractArray{T},
                                                  TS, CB, ST}
    cs, as, rs = SimpleDiffEq._build_tsit5_caches(T)

    !IIP && @assert S <: SArray
    event_last_time = 1
    vector_event_last_time = 0
    last_event_error = zero(eltype(S))

    integ = GPUT5I{IIP, S, T, ST, P, F, TS, CB}(f, copy(u0), copy(u0), copy(u0), t0, t0, t0,
                                                dt,
                                                sign(dt), p, true, tstops, 1, callback,
                                                save_everystep, saveat, 1, 1,
                                                event_last_time,
                                                vector_event_last_time,
                                                last_event_error,
                                                copy(u0), copy(u0), copy(u0), copy(u0),
                                                copy(u0),
                                                copy(u0), copy(u0), cs, as, rs)
end

@inline function gpuatsit5_init(f::F, IIP::Bool, u0::S, t0::T, tf::T, dt::T, p::P,
                                abstol::TOL, reltol::TOL,
                                internalnorm::N, tstops::TS,
                                callback::CB,
                                saveat::ST) where {F, P, S, T, N, TOL, TS, CB, ST}
    cs, as, btildes, rs = SimpleDiffEq._build_atsit5_caches(T)

    !IIP && @assert S <: SArray

    qoldinit = eltype(S)(1e-4)
    event_last_time = 1
    vector_event_last_time = 0
    last_event_error = zero(eltype(S))

    integ = GPUAT5I{IIP, S, T, ST, P, F, N, TOL, typeof(qoldinit), TS, CB}(f, copy(u0),
                                                                           copy(u0),
                                                                           copy(u0), t0, t0,
                                                                           t0,
                                                                           tf, dt,
                                                                           dt,
                                                                           sign(tf - t0), p,
                                                                           true, tstops, 1,
                                                                           callback, false,
                                                                           saveat, 1, 1,
                                                                           event_last_time,
                                                                           vector_event_last_time,
                                                                           last_event_error,
                                                                           copy(u0),
                                                                           copy(u0),
                                                                           copy(u0),
                                                                           copy(u0),
                                                                           copy(u0),
                                                                           copy(u0),
                                                                           copy(u0), cs, as,
                                                                           btildes,
                                                                           rs, qoldinit,
                                                                           abstol,
                                                                           reltol,
                                                                           internalnorm)
end

@inline function gpuvern7_init(f::F, IIP::Bool, u0::S, t0::T, dt::T,
                               p::P, tstops::TS,
                               callback::CB,
                               save_everystep::Bool,
                               saveat::ST) where {F, P, T, S <: AbstractArray{T},
                                                  TS, CB, ST}
    tab = Vern7Tableau(T, T)

    !IIP && @assert S <: SArray
    event_last_time = 1
    vector_event_last_time = 0
    last_event_error = zero(eltype(S))

    integ = GPUV7I{IIP, S, T, ST, P, F, TS, CB, typeof(tab)}(f, copy(u0), copy(u0),
                                                             copy(u0),
                                                             t0, t0, t0, dt,
                                                             sign(dt), p, true, tstops, 1,
                                                             callback,
                                                             save_everystep, saveat, 1, 1,
                                                             event_last_time,
                                                             vector_event_last_time,
                                                             last_event_error,
                                                             copy(u0), copy(u0), copy(u0),
                                                             copy(u0),
                                                             copy(u0),
                                                             copy(u0), copy(u0), copy(u0),
                                                             copy(u0), copy(u0), tab)
end

@inline function gpuavern7_init(f::F, IIP::Bool, u0::S, t0::T, tf::T, dt::T, p::P,
                                abstol::TOL, reltol::TOL,
                                internalnorm::N, tstops::TS,
                                callback::CB,
                                saveat::ST) where {F, P, S, T, N, TOL, TS, CB, ST}
    !IIP && @assert S <: SArray

    tab = Vern7Tableau(T, T)

    qoldinit = eltype(S)(1e-4)
    event_last_time = 1
    vector_event_last_time = 0
    last_event_error = zero(eltype(S))

    integ = GPUAV7I{IIP, S, T, ST, P, F, N, TOL, typeof(qoldinit), TS, CB, typeof(tab)}(f,
                                                                                        copy(u0),
                                                                                        copy(u0),
                                                                                        copy(u0),
                                                                                        t0,
                                                                                        t0,
                                                                                        t0,
                                                                                        tf,
                                                                                        dt,
                                                                                        dt,
                                                                                        sign(tf -
                                                                                             t0),
                                                                                        p,
                                                                                        true,
                                                                                        tstops,
                                                                                        1,
                                                                                        callback,
                                                                                        false,
                                                                                        saveat,
                                                                                        1,
                                                                                        1,
                                                                                        event_last_time,
                                                                                        vector_event_last_time,
                                                                                        last_event_error,
                                                                                        copy(u0),
                                                                                        copy(u0),
                                                                                        copy(u0),
                                                                                        copy(u0),
                                                                                        copy(u0),
                                                                                        copy(u0),
                                                                                        copy(u0),
                                                                                        copy(u0),
                                                                                        copy(u0),
                                                                                        copy(u0),
                                                                                        tab,
                                                                                        qoldinit,
                                                                                        abstol,
                                                                                        reltol,
                                                                                        internalnorm)
end

@inline function gpuvern9_init(f::F, IIP::Bool, u0::S, t0::T, dt::T,
                               p::P, tstops::TS,
                               callback::CB,
                               save_everystep::Bool,
                               saveat::ST) where {F, P, T, S <: AbstractArray{T},
                                                  TS, CB, ST}
    tab = Vern9Tableau(T, T)

    !IIP && @assert S <: SArray
    event_last_time = 1
    vector_event_last_time = 0
    last_event_error = zero(eltype(S))

    integ = GPUV9I{IIP, S, T, ST, P, F, TS, CB, typeof(tab)}(f, copy(u0), copy(u0),
                                                             copy(u0),
                                                             t0, t0, t0, dt,
                                                             sign(dt), p, true, tstops, 1,
                                                             callback,
                                                             save_everystep, saveat, 1, 1,
                                                             event_last_time,
                                                             vector_event_last_time,
                                                             last_event_error,
                                                             copy(u0), copy(u0), copy(u0),
                                                             copy(u0), copy(u0), copy(u0),
                                                             copy(u0), copy(u0),
                                                             copy(u0), copy(u0), tab)
end

@inline function gpuavern9_init(f::F, IIP::Bool, u0::S, t0::T, tf::T, dt::T, p::P,
                                abstol::TOL, reltol::TOL,
                                internalnorm::N, tstops::TS,
                                callback::CB,
                                saveat::ST) where {F, P, S, T, N, TOL, TS, CB, ST}
    !IIP && @assert S <: SArray

    tab = Vern9Tableau(T, T)

    qoldinit = eltype(S)(1e-4)
    event_last_time = 1
    vector_event_last_time = 0
    last_event_error = zero(eltype(S))

    integ = GPUAV9I{IIP, S, T, ST, P, F, N, TOL, typeof(qoldinit), TS, CB, typeof(tab)}(f,
                                                                                        copy(u0),
                                                                                        copy(u0),
                                                                                        copy(u0),
                                                                                        t0,
                                                                                        t0,
                                                                                        t0,
                                                                                        tf,
                                                                                        dt,
                                                                                        dt,
                                                                                        sign(tf -
                                                                                             t0),
                                                                                        p,
                                                                                        true,
                                                                                        tstops,
                                                                                        1,
                                                                                        callback,
                                                                                        false,
                                                                                        saveat,
                                                                                        1,
                                                                                        1,
                                                                                        event_last_time,
                                                                                        vector_event_last_time,
                                                                                        last_event_error,
                                                                                        copy(u0),
                                                                                        copy(u0),
                                                                                        copy(u0),
                                                                                        copy(u0),
                                                                                        copy(u0),
                                                                                        copy(u0),
                                                                                        copy(u0),
                                                                                        copy(u0),
                                                                                        copy(u0),
                                                                                        copy(u0),
                                                                                        tab,
                                                                                        qoldinit,
                                                                                        abstol,
                                                                                        reltol,
                                                                                        internalnorm)
end
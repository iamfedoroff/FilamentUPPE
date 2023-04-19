module FilamentUPPE


import Adapt: @adapt_structure, adapt
import AnalyticSignals: rsig2asig!, rsig2aspec!
import CUDA: CUDA, @cuda, launch_configuration, CuArray, CuVector, synchronize,
             threadIdx, blockIdx, blockDim, gridDim
import Dates: now, canonicalize, CompoundPeriod
import FFTW: fftfreq, plan_fft!, ifftshift, ifft!
import HankelTransforms: dhtfreq, plan_dht
import ODEIntegrators: Problem, Integrator, step, step!,
                       RK2, RK3, SSPRK3, SSP4RK3, RK4, Tsit5, ATsit5
import Printf: @printf
import TimerOutputs: @timeit, reset_timer!, print_timer

using PhysicalConstants.CODATA2018
const C0 = SpeedOfLightInVacuum.val
const EPS0 = VacuumElectricPermittivity.val
const MU0 = VacuumMagneticPermeability.val
const QE = ElementaryCharge.val
const ME = ElectronMass.val
const HBAR = ReducedPlanckConstant.val

using FilamentBase
import FilamentBase: radius, linterp

import FilamentPlasma: Plasma, solve!


# ODEIntegrators
export RK2, RK3, SSPRK3, SSP4RK3, RK4, Tsit5, ATsit5

# FilamentBase
export CPU, GPU, GridR, GridT, GridRT, GridXY, GridXYT, Field, Medium,
       refractive_index, k_func, k1_func, k2_func, phase_velocity,
       group_velocity, diffraction_length, dispersion_length, absorption_length,
       chi1_func, chi3_func, critical_power, nonlinearity_length,
       selffocusing_length

# FilamentPlasma:
export Plasma, solve!

# FilamentUPPE
export Kerr, Raman, Current, CurrentLosses, Model, model_run!

CUDA.allowscalar(false)


include("nonlinearities/Nonlinearities.jl")
include("models/Models.jl")


function phi_kerr(model)
    (; zu, field, medium) = model
    (; Eu, w0) = field
    (; permeability) = medium

    mu = permeability(w0)
    k0 = k_func(medium, w0)
    chi3 = chi3_func(medium, w0)

    QQ0 = MU0 * mu * w0^2 / (2 * real(k0)) * zu / Eu
    Rk0 = EPS0 * chi3 * 3 / 4 * Eu^3
    return QQ0 * abs(real(Rk0))
end


function phi_plasma(model)
    (; zu, field, medium, plasma) = model
    (; Eu, w0) = field
    (; permeability) = medium
    (; neu) = plasma

    mu = permeability(w0)
    k0 = k_func(medium, w0)

    QQ0 = MU0 * mu * w0^2 / (2 * real(k0)) * zu / Eu
    Rp0 = 1im / w0 * QE^2 / ME / (-1im * w0) * neu * Eu
    return QQ0 * abs(real(Rp0))
end


end

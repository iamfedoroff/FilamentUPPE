struct ModelT{T, TG, TF, TM, G, TFFT, TK, TC} <: Model
    # units of propagation distance:
    zu :: T
    # grid & field:
    grid :: TG
    field :: TF
    medium :: TM
    # guards:
    guard_t :: G
    guard_w :: G
    # spectral parameters:
    FFT :: TFFT
    w :: TK
    KK :: TC
end

@adapt_structure ModelT


function Model(grid::GridT, field, medium; zu=1, tguard=0, lcut=0.2e-6)
    (; Nt, tu, dt, t) = grid
    (; w0, E) = field

    FFT = plan_fft!(E)
    rsig2asig!(E, FFT)   # real signal -> analytic signal

    wu = 1 / tu
    w = 2*pi * fftfreq(Nt, 1/dt)

    vf = group_velocity(medium, w0)   # frame velocity
    KK = zeros(ComplexF64, Nt)
    for it=1:Nt
        k = k_func(medium, w[it] * wu)
        KK[it] = (sqrt(k^2 + 0im) - abs(w[it] * wu) / vf) * zu
    end

    guard_t = guard(t, tguard; shape=:both)

    wcut = 2*pi * C0 / lcut
    guard_w = @. exp(-(w * wu / wcut)^40)

    return ModelT(zu, grid, field, medium, guard_t, guard_w, FFT, w, KK)
end


function model_step!(model::ModelT, z, dz)
    (; field, FFT, KK, guard_t) = model
    (; E) = field
    FFT \ E   # time -> frequency [exp(-i*w*t)]
    @. E = E * exp(1im * KK * dz)
    FFT * E   # frequency -> time [exp(-i*w*t)]
    mulvec!(E, guard_t; dim=1)
    return nothing
end


function model_run!(
    model::ModelT; arch=CPU(), prefix="results/", z=0, zmax, dz0, dzhdf,
)
    model = adapt(arch, model)

    (; zu, grid, field) = model
    (; Nt, t, dt) = grid
    (; E) = field

    outtxt = OutputTXT(prefix * "out.txt", grid, field; zu)
    outhdf = OutputHDF(prefix * "out.hdf", grid, field; zu, z, dzhdf, func=real)

    I = adapt(arch, zeros(Nt))

    while z <= zmax + dz0
        @. I = abs2(E)

        Imax = maximum(I)
        nemax = 0
        tau = radius(t, collect(I))
        F = sum(abs2, E) * dt
        @printf("%18.12e %18.12e %18.12e\n", z, Imax, nemax)
        writetxt(outtxt, (z, Imax, nemax, tau, F))
        writehdf(outhdf, z)

        z += dz0

        model_step!(model, z, dz0)
    end
    return nothing
end

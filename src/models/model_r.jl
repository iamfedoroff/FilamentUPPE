struct ModelR{T, TG, TF, TM, G, TDHT, TK, TN} <: Model
    # units of propagation distance:
    zu :: T
    # grid & field:
    grid :: TG
    field :: TF
    medium :: TM
    # guards:
    guard_r :: G
    # spectral parameters:
    DHT :: TDHT
    KK :: TK
    # plasma & nonlinearities:
    nonlinearities :: TN
end

@adapt_structure ModelR


function Model(
    grid::GridR, field, medium;
    zu=1,
    rguard=0,
    kparaxial=false,
    nonlinearities=nothing,
    vf=nothing,
)
    (; Nr, ru, rmax, r) = grid
    (; Eu, w0, E) = field

    DHT = plan_dht(rmax, E; save=false)

    ku = 1 / ru
    k = 2*pi * dhtfreq(rmax, Nr)

    if isnothing(vf)
        vf = group_velocity(medium, w0)   # frame velocity
    end
    KK = zeros(ComplexF64, Nr)
    for ir=1:Nr
        kt = k[ir] * ku
        KK[ir] = KK_func(medium, w0, kt, kparaxial) * zu - w0 / vf * zu
    end

    guard_r = guard(r, rguard; shape=:right)

    return ModelR(zu, grid, field, medium, guard_r, DHT, KK, nonlinearities)
end


# function qfunc!(dE, E, p, z)
#     model, F = p
#     (; FFT, QQ, plasma, nonlinearities, guard_w) = model
#     # (; ne) = plasma

#     FFT * E   # frequency -> time [exp(-i*w*t)]

#     @. dE = 0
#     for nl in nonlinearities
#         nl.func!(F, E, nl.p, z, plasma)
#         rsig2aspec!(F, FFT)   # real signal -> analytic spectrum
#         mulvec!(F, nl.R; dim=2)
#         @. dE += F
#     end
#     @. dE *= 1im * QQ
#     mulvec!(dE, guard_w; dim=2)

#     FFT \ E   # time -> frequency [exp(-i*w*t)]
#     return nothing
# end


function model_step!(model::ModelR, qinteg, z, dz)
    (; field, DHT, KK, guard_r) = model
    (; E) = field

    # @timeit "Q step" begin
    #     if !isnothing(nonlinearities)
    #         step!(qinteg, E, z, dz)
    #     end
    #     synchronize()
    # end
    @timeit "K step" begin
        DHT * E
        @. E = E * exp(1im * KK * dz)
        DHT \ E
        synchronize()
    end
    @timeit "guards" begin
        mulvec!(E, guard_r; dim=1)
        synchronize()
    end

    return nothing
end


function model_run!(
    model::ModelR; arch=CPU(), prefix="results/", z=0, zmax, dz0, dzhdf,
    phimax=pi/100, Istop=Inf, alg=RK3(),
)
    model = adapt(arch, model)

    (; zu, grid, field, nonlinearities) = model
    (; r, dr) = grid
    (; E) = field

    outtxt = OutputTXT(prefix * "out.txt", grid, field; zu)
    outhdf = OutputHDF(prefix * "out.hdf", grid, field; zu, z, dzhdf)

    if !isnothing(nonlinearities)
        phik = phi_kerr(model)
        prob = Problem(qfunc!, E, (model, zero(E)))
        qinteg = Integrator(prob, alg)
    else
        qinteg = nothing
    end

    zfirst = true

    stime = now()

    while z <= zmax + dz0
        @timeit "observables" begin
            Imax = maximum(abs2, E)
            rad = 0
            P = 2*pi * sum(abs2.(E) .* r .* dr)
            synchronize()
        end

        @timeit "outputs" begin
            @printf("%18.12e %18.12e\n", z, Imax)
            writetxt(outtxt, (z, Imax, rad, P))
            writehdf(outhdf, z)
            synchronize()
        end

        # adaptive z step:
        if !isnothing(nonlinearities)
            dzk = phimax / (phik * Imax)
        else
            dzk = Inf
        end
        dz = min(dz0, dzk)
        z += dz

        @timeit "model step" begin
            model_step!(model, qinteg, z, dz)
            synchronize()
        end

        if zfirst
            reset_timer!()
            zfirst = false
        end

        if Imax > Istop
            @warn "Imax >= Istop"
            break
        end
    end

    print_timer()

    etime = now()
    ttime = canonicalize(CompoundPeriod(etime - stime))
    message = "Start time: $(stime)\n" *
              "End time:   $(etime)\n" *
              "Run time:   $(ttime)"
    println(message)

    return nothing
end

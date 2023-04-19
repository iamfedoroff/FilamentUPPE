struct ModelXYT{T, TG, TF, TM, G, TFFT1, TFFT2, TK, TP, TN} <: Model
    # units of propagation distance:
    zu :: T
    # grid & field:
    grid :: TG
    field :: TF
    medium :: TM
    # guards:
    guard_x :: G
    guard_y :: G
    guard_t :: G
    guard_w :: G
    # spectral parameters:
    FFTxy :: TFFT1
    FFTt :: TFFT2
    KK :: TK
    QQ :: TK
    # plasma & nonlinearities:
    plasma :: TP
    nonlinearities :: TN
end

@adapt_structure ModelXYT


function Model(
    grid::GridXYT, field, medium;
    zu=1,
    xguard=0,
    yguard=0,
    tguard=0,
    lcut=0.2e-6,
    kparaxial=false,
    qparaxial=true,
    plasma=nothing,
    nonlinearities=nothing,
    vf=nothing,
)
    (; Nx, Ny, Nt, xu, yu, tu, dx, dy, dt, x, y, t) = grid
    (; Eu, w0, E) = field

    FFTxy = plan_fft!(E, [1,2])

    FFTt = plan_fft!(E, [3])
    rsig2asig!(E, FFT)   # real signal -> analytic signal

    kxu = 1 / xu
    kyu = 1 / yu
    wu = 1 / tu
    kx = 2*pi * fftfreq(Nx, 1/dx)
    ky = 2*pi * fftfreq(Ny, 1/dy)
    w = 2*pi * fftfreq(Nt, 1/dt)

    if isnothing(vf)
        vf = group_velocity(medium, w0)   # frame velocity
    end
    KK = zeros(ComplexF64, (Nx, Ny, Nt))
    QQ = zeros(ComplexF64, (Nx, Ny, Nt))
    for it=1:Nt, iy=1:Ny, ix=1:Nx
        kt = sqrt((kx[ix] * kxu)^2 + (ky[iy] * kyu)^2)
        ww = w[it] * wu
        KK[ix,iy,it] = KK_func(medium, ww, kt, kparaxial) * zu - abs(ww) / vf * zu
        QQ[ix,iy,it] = QQ_func(medium, ww, kt, qparaxial) * zu / Eu
    end

    guard_x = guard(x, xguard; shape=:both)
    guard_y = guard(y, yguard; shape=:both)
    guard_t = guard(t, tguard; shape=:both)

    wcut = 2*pi * C0 / lcut
    guard_w = @. exp(-(w * wu / wcut)^40)

    return ModelXYT(
        zu, grid, field, medium, guard_x, guard_y, guard_t, guard_w, FFTxy,
        FFTx, KK, QQ, plasma, nonlinearities,
    )
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


# function model_step!(model::ModelRT, qinteg, z, dz)
#     (; field, plasma, nonlinearities, DHT, FFT, KK, guard_r, guard_t) = model
#     (; E) = field

#     @timeit "plasma" begin
#         if !isnothing(plasma)
#             solve!(plasma, E)
#         end
#         synchronize()
#     end
#     @timeit "time -> freq" begin
#         FFT \ E   # time -> frequency [exp(-i*w*t)]
#         synchronize()
#     end
#     @timeit "Q step" begin
#         if !isnothing(nonlinearities)
#             step!(qinteg, E, z, dz)
#         end
#         synchronize()
#     end
#     @timeit "K step" begin
#         DHT * E
#         @. E = E * exp(1im * KK * dz)
#         DHT \ E
#         synchronize()
#     end
#     @timeit "freq -> time" begin
#         FFT * E   # frequency -> time [exp(-i*w*t)]
#         synchronize()
#     end
#     @timeit "guards" begin
#         mulvec!(E, guard_r; dim=1)
#         mulvec!(E, guard_t; dim=2)
#         synchronize()
#     end

#     return nothing
# end


# function model_run!(
#     model::ModelRT; arch=CPU(), prefix="results/", z=0, zmax, dz0, dzhdf,
#     phimax=pi/100, Istop=Inf, alg=RK3(),
# )
#     model = adapt(arch, model)

#     (; zu, grid, field, plasma, nonlinearities, FFT) = model
#     (; Nr, Nt, r, t, dr, dt) = grid
#     (; E) = field

#     Nwr = iseven(Nt) ? div(Nt,2) : div(Nt+1,2)
#     Fr, Ft, Si, ner = zeros(Nr), zeros(Nt), zeros(Nwr+1), zeros(Nr)
#     Fr, Ft, Si, ner = (adapt(arch, x) for x in (Fr, Ft, Si, ner))
#     zvars = Dict("Fr" => Fr, "Ft" => Ft, "Si" => Si, "ne" => ner)

#     outtxt = OutputTXT(prefix * "out.txt", grid, field; zu)
#     outhdf = OutputHDF(prefix * "out.hdf", grid, field; zu, z, dzhdf, func=real, zvars)

#     if !isnothing(plasma)
#         (; ne) = plasma
#         solve!(plasma, E)
#         phip = phi_plasma(model)
#     end
#     if !isnothing(nonlinearities)
#         phik = phi_kerr(model)
#         prob = Problem(qfunc!, E, (model, zero(E)))
#         qinteg = Integrator(prob, alg)
#     else
#         qinteg = nothing
#     end

#     zfirst = true

#     stime = now()

#     while z <= zmax + dz0
#         @timeit "observables" begin
#             Fr .= sum(abs2, E; dims=2) * dt
#             Ft .= 2*pi * vec(sum(abs2.(E) .* r .* dr; dims=1))

#             FFT \ E   # time -> frequency [exp(-i*w*t)]
#             Si .= 2*pi * dt * Nt * ifftshift(vec(sum(abs2.(E) .* r .* dr, dims=1)))[Nwr:end]
#             FFT * E   # frequency -> time [exp(-i*w*t)]

#             if !isnothing(plasma)
#                 ner .= @views ne[:,end]
#                 nemax = maximum(ne)
#             else
#                 nemax = 0
#             end

#             Imax = maximum(abs2, E)
#             Fmax = maximum(Fr)
#             rad = 2 * radius(collect(r), collect(Fr))
#             tau = radius(t, collect(Ft))
#             W = sum(Ft) * dt

#             synchronize()
#         end

#         @timeit "outputs" begin
#             @printf("%18.12e %18.12e %18.12e\n", z, Imax, nemax)
#             writetxt(outtxt, (z, Imax, Fmax, nemax, rad, tau, W))
#             writehdf(outhdf, z)
#             synchronize()
#         end

#         # adaptive z step:
#         if !isnothing(plasma)
#             dzp = phimax / (phip * nemax)
#         else
#             dzp = Inf
#         end
#         if !isnothing(nonlinearities)
#             dzk = phimax / (phik * Imax)
#         else
#             dzk = Inf
#         end
#         dz = min(dz0, dzp, dzk)
#         z += dz

#         @timeit "model step" begin
#             model_step!(model, qinteg, z, dz)
#             synchronize()
#         end

#         if zfirst
#             reset_timer!()
#             zfirst = false
#         end

#         if Imax > Istop
#             @warn "Imax >= Istop"
#             break
#         end
#     end

#     print_timer()

#     etime = now()
#     ttime = canonicalize(CompoundPeriod(etime - stime))
#     message = "Start time: $(stime)\n" *
#               "End time:   $(etime)\n" *
#               "Run time:   $(ttime)"
#     println(message)

#     return nothing
# end

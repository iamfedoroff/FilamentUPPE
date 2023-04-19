struct ModelXY{T, TG, TF, TM, G, TFFT, TK} <: Model
    # units of propagation distance:
    zu :: T
    # grid & field:
    grid :: TG
    field :: TF
    medium :: TM
    # guards:
    guard_x :: G
    guard_y :: G
    # spectral parameters:
    FFT :: TFFT
    KK :: TK
end

@adapt_structure ModelXY


function Model(
    grid::GridXY, field, medium; zu=1, xguard=0, yguard=0, kparaxial=false,
)
    (; Nx, Ny, xu, yu, dx, dy, x, y) = grid
    (; w0, E) = field

    FFT = plan_fft!(E)

    kxu = 1 / xu
    kyu = 1 / yu
    kx = 2*pi * fftfreq(Nx, 1/dx)
    ky = 2*pi * fftfreq(Ny, 1/dy)

    k0 = k_func(medium, w0)
    KK = zeros(ComplexF64, (Nx, Ny))
    for iy=1:Ny, ix=1:Nx
        kt2 = (kx[ix] * kxu)^2 + (ky[iy] * kyu)^2
        if kparaxial
            KK[ix,iy] = (k0 - kt2 / (2 * k0)) * zu
        else
            KK[ix,iy] = sqrt(k0^2 - kt2 + 0im) * zu
        end
        KK[ix,iy] -= k0 * zu   # shift KK by k0 for higher precesion
    end

    guard_x = guard(x, xguard; shape=:both)
    guard_y = guard(y, yguard; shape=:both)

    return ModelXY(zu, grid, field, medium, guard_x, guard_y, FFT, KK)
end


function model_step!(model::ModelXY, z, dz)
    (; field, FFT, KK, guard_x, guard_y) = model
    (; E) = field
    FFT * E
    @. E = E * exp(1im * KK * dz)
    FFT \ E
    mulvec!(E, guard_x; dim=1)
    mulvec!(E, guard_y; dim=2)
    return nothing
end


function model_run!(
    model::ModelXY; arch=CPU(), prefix="results/", z=0, zmax, dz0, dzhdf,
    Istop=Inf,
)
    model = adapt(arch, model)

    (; zu, grid, field) = model
    (; Nx, Ny, x, y, dx, dy) = grid
    (; E) = field

    Ix, Iy = zeros(Nx), zeros(Ny)
    Ix, Iy = (adapt(arch, x) for x in (Ix, Iy))
    zvars = Dict("Ix" => Ix, "Iy" => Iy)

    outtxt = OutputTXT(prefix * "out.txt", grid, field; zu)
    outhdf = OutputHDF(prefix * "out.hdf", grid, field; zu, z, dzhdf, zvars)

    zfirst = true

    stime = now()

    while z <= zmax + dz0
        @timeit "observables" begin
            Imax = maximum(abs2, E)
            @. Ix = abs2(E[:,div(Ny,2)])
            @. Iy = abs2(E[div(Nx,2),:])
            radx = 0
            rady = 0
            P = sum(abs2, E) * dx * dy
            synchronize()
        end

        @timeit "outputs" begin
            @printf("%18.12e %18.12e\n", z, Imax)
            writetxt(outtxt, (z, Imax, radx, rady, P))
            writehdf(outhdf, z)
            synchronize()
        end

        z += dz0

        @timeit "model step" begin
            model_step!(model, z, dz0)
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

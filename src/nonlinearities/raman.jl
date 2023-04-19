function Raman_func!(F, E, p, z, plasma)
    Hraman, FFT = p
    @. F = real(E)^2
    # convolution!(F, PT, Hraman)
    FFT \ F   # time -> frequency [exp(-i*w*t)]
    mulvec!(F, Hraman; dim=ndims(F))   # assume time along the last dimension
    FFT * F    # frequency -> time [exp(-i*w*t)]
    @. F = F * real(E)
    return nothing
end


function Raman(grid, field, medium; raman_response, rfrac=1)
    (; tu, Nt, tmin, tmax, t, dt) = grid
    (; Eu, w0, E) = field

    chi3 = chi3_func(medium, w0)
    R = rfrac * EPS0 * chi3 * Eu^3

    # For assymetric grids, where abs(tmin) != tmax, we need tshift to put H(t)
    # into the grid center (see "circular convolution"):
    tshift = tmin + (tmax - tmin) / 2
    Hraman = @. raman_response((t - tshift) * tu)
    Hraman = Hraman * tu
    Hraman = @. Hraman + 0im   # real -> complex

    # The correct way to calculate spectrum which matches theory:
    #    S = ifftshift(E)   # compensation of the spectrum oscillations
    #    S = ifft(S) * len(E) * dt   # normalization
    #    S = fftshift(S)   # recovery of the proper array order
    Hraman = ifftshift(Hraman)
    ifft!(Hraman)   # time -> frequency [exp(-i*w*t)]
    @. Hraman = Hraman * Nt * dt

    FFT = plan_fft!(E, [2])

    return Nonlinearity(R, Raman_func!, (Hraman, FFT))
end

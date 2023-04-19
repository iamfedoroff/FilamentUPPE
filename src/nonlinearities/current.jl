function Current_func!(F, E, p, z, plasma)
    @. F = plasma.ne * real(E)
    return nothing
end


function Current(grid, field, medium; plasma)
    (; Nt, tu, dt) = grid
    (; Eu) = field
    (; neu, mr, nuc) = plasma

    wu = 1 / tu
    w = 2*pi * fftfreq(Nt, 1/dt)
    MR = mr * ME

    R = zeros(ComplexF64, Nt)
    for it=1:Nt
        ww = w[it] * wu
        if ww !=0
            sigma = QE^2 / MR * (nuc + 1im * ww) / (nuc^2 + ww^2)
            R[it] = 1im / ww * sigma * neu * Eu
        end
    end

    return Nonlinearity(R, Current_func!, ())
end

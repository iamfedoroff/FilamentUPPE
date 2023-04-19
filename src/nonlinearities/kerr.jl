function Kerr_func!(F, E, p, z, plasma)
    @. F = real(E)^3
    return nothing
end


function Kerr(grid, field, medium; rfrac=0)
    (; Eu, w0) = field
    chi3 = chi3_func(medium, w0)
    R = (1 - rfrac) * EPS0 * chi3 * Eu^3
    return Nonlinearity(R, Kerr_func!, ())
end

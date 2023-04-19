function CurrentLosses_func!(F, E, p, z, plasma)
    @. F = abs2(E)
    inverse!(F)
    @. F = plasma.kdne * F * real(E)
    return nothing
end


function CurrentLosses(grid, field, medium; plasma)
    (; Nt, tu, dt) = grid
    (; Eu, w0) = field
    (; neu) = plasma

    wu = 1 / tu
    w = 2*pi * fftfreq(Nt, 1/dt)

    R = zeros(ComplexF64, Nt)
    for it=1:Nt
        ww = w[it] * wu
        if ww !=0
            R[it] = 1im / ww * HBAR * w0 * neu / (tu * Eu)
        end
    end

    return Nonlinearity(R, CurrentLosses_func!, ())
end


function inverse!(A)
    for i in eachindex(A)
        if real(A[i]) >= 1e-30
            A[i] = 1 / A[i]
        else
            A[i] = 0
        end
    end
    return nothing
end


function inverse!(A::CuArray)
    N = length(A)
    @krun N inverse_kernel!(A)
    return nothing
end
function inverse_kernel!(A)
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    for i=id:stride:length(A)
        if real(A[i]) >= 1f-30
            A[i] = 1 / A[i]
        else
            A[i] = 0
        end
    end
    return nothing
end

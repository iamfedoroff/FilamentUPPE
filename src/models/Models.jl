abstract type Model end

include("model_r.jl")
include("model_t.jl")
include("model_rt.jl")
include("model_xy.jl")
include("model_xyt.jl")


function KK_func(medium, w, kt, paraxial)
    k = k_func(medium, w)
    KK = 0
    if paraxial
        if k != 0
            KK = k - kt^2 / (2 * k)
        end
    else
        KK = sqrt(k^2 - kt^2 + 0im)
    end
    return KK
end


function QQ_func(medium, w, kt, paraxial)
    mu = medium.permeability(w)
    k = k_func(medium, w)
    QQ = 0
    if paraxial
        if k !=0
            QQ = MU0 * mu * w^2 / (2 * k)
        end
    else
        kz = sqrt(k^2 - kt^2 + 0im)
        if kz !=0
            QQ = MU0 * mu * w^2 / (2 * kz)
        end
    end
    return QQ
end

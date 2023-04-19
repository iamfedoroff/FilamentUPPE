struct Nonlinearity{T,F,P}
    R :: T
    func! :: F
    p :: P
end

@adapt_structure Nonlinearity


include("kerr.jl")
include("raman.jl")
include("current.jl")
include("current_losses.jl")

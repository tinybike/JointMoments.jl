normalize{T<:Real}(w::Array{T}) = vec(w) / sum(w)

function coskewness{T<:Real}(data::Array{T}, w::Array{T}; standardize=false, bias=0)
    if sum(w) != 1
        w = normalize(w)
    end
    length(w) == size(data, 2) || throw(DimensionMismatch("Inconsistent array lengths."))
    w' * coskew(data, standardize=standardize, flatten=true, bias=bias) * kron(w, w)
end

function cokurtosis{T<:Real}(data::Array{T}, w::Array{T}; standardize=false, bias=0)
    if sum(w) != 1
        w = normalize(w)
    end
    length(w) == size(data, 2) || throw(DimensionMismatch("Inconsistent array lengths."))
    w' * cokurt(data, standardize=standardize, flatten=true, bias=bias) * kron(kron(w, w), w)
end

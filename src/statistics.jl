# Standard deviation of a vector without Bessel's correction
function Base.std{T<:Real}(data::Vector{T}, avg::Real, n::Int)
    s = 0.0
    for i = 1:n
        @inbounds z = data[i] - avg
        s += z * z
    end
    sqrt(s / n)
end

# Per-column standard deviation of a matrix without Bessel's correction
Base.std{T<:Real}(data::Matrix{T}, avg::Vector{T}, rows::Int, cols::Int) = 
    vec([std(data[:,j], avg[j], rows) for j in 1:cols])

Base.std{T<:Real}(data::Matrix{T}, avg::Matrix{T}, rows::Int, cols::Int) = 
    std(data, vec(avg), rows, cols)

normalize{T<:Real}(v::Vector{T}) = vec(v) / sum(v)

normalize{T<:Real}(v::Matrix{T}) = normalize(vec(v))

function coskewness{T<:Real}(data::Matrix{T}, w::Vector{T}; standardize=false, bias=0)
    if sum(w) != 1
        w = normalize(w)
    end
    length(w) == size(data, 2) || throw(DimensionMismatch("Inconsistent array lengths."))
    first(w' * coskew(data, standardize=standardize, flatten=true, bias=bias) * kron(w, w))
end

coskewness{T<:Real}(data::Matrix{T}; standardize=false, bias=0) =
    coskewness(data, ones(size(data, 2)), standardize=standardize, bias=bias)

coskewness{T<:Real}(data::Matrix{T}, w::Matrix{T}; standardize=false, bias=0) =
    coskewness(data, vec(w), standardize=standardize, bias=bias)

function cokurtosis{T<:Real}(data::Matrix{T}, w::Vector{T}; standardize=false, bias=0)
    if sum(w) != 1
        w = normalize(w)
    end
    length(w) == size(data, 2) || throw(DimensionMismatch("Inconsistent array lengths."))
    first(w' * cokurt(data, standardize=standardize, flatten=true, bias=bias) * kron(kron(w, w), w))
end

cokurtosis{T<:Real}(data::Matrix{T}; standardize=false, bias=0) =
    cokurtosis(data, ones(size(data, 2)), standardize=standardize, bias=bias)

cokurtosis{T<:Real}(data::Matrix{T}, w::Matrix{T}; standardize=false, bias=0) =
    cokurtosis(data, vec(w), standardize=standardize, bias=bias)

# Weighted row-replication
function replicate{T<:Real}(data::Matrix{T}, w::Vector{T})
    num_samples, num_signals = size(data)
    length(w) == num_samples ||
        throw(DimensionMismatch("Inconsistent array lengths."))
    replicated = zeros(int(sum(w)), num_signals)
    j = 1
    for i = 1:num_samples
        for k = j:(j+int(w[i])-1)
            replicated[k,:] = data[i,:]
        end
        j += int(w[i])
    end
    replicated
end

# Recombine over weights
function recombine{T<:Real}(v::Vector{T}, w::Vector{T})
    num_weights = length(w)
    recombined = zeros(num_weights)
    j = 1
    for i = 1:num_weights
        recombined[i] = v[j] * w[i]
        j += int(w[i])
    end
    recombined
end

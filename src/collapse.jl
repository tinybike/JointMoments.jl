# Collapse tensor to vector: internal sum over fibers
function collapse{T<:Real}(data::Matrix{T},
                           order::Int;
                           standardize::Bool=false,
                           bias::Int=0)
    # Center/whiten data
    cntr, num_samples, num_signals = center(data;
                                            standardize=standardize,
                                            bias=bias)
    # Sum over fibers
    vec(sum(cntr, 2)'.^(order - 1) * cntr) / (num_samples - bias)
end

# Weighted tensor collapse
function collapse{T<:Real}(data::Matrix{T},
                           w::Vector{T},
                           order::Int;
                           standardize::Bool=false,
                           axis::Int=1,
                           bias::Int=0)
    # Center and weight data
    cntr, num_samples, num_signals = center(data,
                                            normalize(w);
                                            axis=axis,
                                            standardize=standardize,
                                            bias=bias)
    # Sum over fibers
    vec(sum(cntr, 2)'.^(order - 1) * cntr) / (num_samples - bias)
end

# Collapse tensor to vector: internal sum over fibers
function collapse{T<:Real}(data::Matrix{T};
                           order::Int=2,
                           standardize::Bool=false,
                           bias::Int=0,
                           normalized::Bool=false)
    order > 1 || throw(DomainError())
    
    # Center/whiten data
    cntr, num_samples, num_signals = center(data;
                                            standardize=standardize,
                                            bias=bias)
    # Sum over fibers
    collapsed = vec(sum(cntr, 2)'.^(order - 1) * cntr) / (num_samples - bias)
    (normalized) ? normalize(collapsed) : collapsed
end

# Weighted tensor collapse
function collapse{T<:Real}(data::Matrix{T},
                           w::Vector{T};
                           order::Int=2,
                           axis::Int=1,
                           standardize::Bool=false,
                           bias::Int=0,
                           normalized::Bool=false)
    order > 1 || throw(DomainError())
    if axis == 1
        cntr, num_samples, num_signals = center(data,
                                                normalize(w);
                                                standardize=standardize,
                                                bias=bias)
        vec(sum(cntr, 2)'.^(order - 1) * cntr) / (num_samples - bias)
    else
        collapsed = collapse(
            w'.*data';
            order=order,
            standardize=standardize,
            bias=bias,
        )
        (normalized) ? normalize(collapsed) : collapsed
    end
end

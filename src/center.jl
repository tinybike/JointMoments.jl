flip(v, axis) = (axis == 1) ? v : v'

# Center, standardize, and/or weight data matrix
function center{T<:Real}(data::Matrix{T};
                         standardize::Bool=false,
                         bias::Int=0)
    num_samples, num_signals = size(data)
    @inbounds begin
        avgs = vec(mean(data, 1))
        cntr = data .- avgs'

        # Standardized moments: divide by the per-signal standard deviation
        if standardize
            if bias == 1
                cntr ./= std(data, 1)
            else
                cntr ./= std(data, avgs', num_samples, num_signals)'
            end
        end
    end
    (cntr, num_samples, num_signals)
end

# Center/whiten weighted data
function center{T<:Real}(data::Matrix{T},
                         w::Vector{T};
                         standardize::Bool=false,
                         axis::Int=1,
                         bias::Int=0)
    num_samples, num_signals = size(data)
    w = flip(normalize(w), axis)

    # Weighted detrending
    @inbounds begin
        weighted_data = data .* w
        avgs = mean(weighted_data, 1)
        cntr = data .- avgs
        if standardize
            if bias == 1
                cntr ./= std(cntr, axis)
            else
                cntr ./= std(cntr, mean(cntr, 1)', num_samples, num_signals)'
            end
        end
    end
    (cntr, num_samples, num_signals)
    # (cntr .* w, num_samples, num_signals)
end

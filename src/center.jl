# Center and/or whiten data matrix
function center{T<:Real}(data::Matrix{T};
                         standardize::Bool=false,
                         bias::Int=0)
    num_samples, num_signals = size(data)
    @inbounds begin
        avgs = mean(data, 1)
        cntr = data .- avgs

        # Standardized moments: divide by the per-signal standard deviation
        if standardize
            stddev = (bias == 1) ? std(data, 1) :
                std(data, vec(avgs), num_samples, num_signals)'
            all(stddev .!= 0) ||
                error("Cannot standardize: variance = 0")
            cntr ./= stddev
        end
    end
    (cntr, num_samples, num_signals)
end

# Centering/whitening for weighted data
function center{T<:Real}(data::Matrix{T},
                         w::Vector{T};
                         standardize::Bool=false,
                         bias::Int=0)
    num_samples, num_signals = size(data)
    length(w) == num_samples || throw(DimensionMismatch("Inconsistent array lengths."))

    # Weighted detrending
    @inbounds begin
        avgs = mean(data, weights(w), 1)
        cntr = data .- avgs
        if standardize
            if bias == 1
                cntr ./= std(data, 1)
            else
                cntr ./= std(data, vec(avgs), num_samples, num_signals)'
            end
        end
    end
    (cntr, num_samples, num_signals)
end

# Standard deviation of a centered vector (adjustable bias)
function _std{T<:Real}(data::Vector{T}, n::Int, bias::Int)
    if bias == 0
        s = 0.0
        for i = 1:n
            @inbounds s += data[i]^2
        end
        sqrt(s / n)
    elseif bias == 1
        std(data)
    end
end

# Standard deviation of a vector (adjustable bias)
function _std{T<:Real}(data::Vector{T}, avg::Real, n::Int, bias::Int)
    if bias == 0
        s = 0.0
        for i = 1:n
            @inbounds z = data[i] - avg
            s += z * z
        end
        sqrt(s / n)
    elseif bias == 1
        std(data)
    end
end

# Per-column standard deviation of a centered matrix (adjustable bias)
function _std{T<:Real}(data::Matrix{T}, rows::Int, cols::Int, bias::Int)
    if bias == 0
        stds = (Real)[]
        for j = 1:cols
            push!(stds, _std(data[:,j], rows, bias))
        end
        stds
    elseif bias == 1
        vec(std(data, 1))
    end
end

# Per-column standard deviation of a matrix (adjustable bias)
function _std{T<:Real}(data::Matrix{T}, avg::Vector{T}, rows::Int, cols::Int, bias::Int)
    if bias == 0
        stds = (Real)[]
        for j = 1:cols
            push!(stds, _std(data[:,j], avg[j], rows, bias))
        end
        stds
    elseif bias == 1
        vec(std(data, 1))
    end
end

# Covariance matrix (for testing)
function _cov{T<:Real}(data::Matrix{T}; bias::Int=0, dense::Bool=true)
    num_samples, num_signals = size(data)
    adj = num_samples - bias
    @inbounds begin
        avgs = vec(mean(data, 1))
        cntr = data .- avgs'
    end
    tensor = zeros(num_signals, num_signals)

    # Dense matrix (all values are calculated, including duplicates)
    if dense
        @simd for i = 1:num_signals
            @simd for j = 1:num_signals
                @inbounds tensor[i,j] = sum(cntr[:,i].*cntr[:,j]) / adj
            end
        end

    # Lower triangular matrix (duplicate values not calculated)
    # Convert to dense matrix using:
    #   tensor + tensor' - diagm(diag(tensor)))
    else
        @simd for i = 1:num_signals
            @simd for j = 1:i
                @inbounds tensor[i,j] = sum(cntr[:,i].*cntr[:,j]) / adj
            end
        end
    end
    return tensor
end

# Coskewness tensor (third-order)
function coskew{T<:Real}(data::Matrix{T};
                         standardize::Bool=false,
                         flatten::Bool=false,
                         bias::Int=0,
                         dense::Bool=true)
    num_samples, num_signals = size(data)
    adj = num_samples - bias
    @inbounds begin
        avgs = vec(mean(data, 1))
        cntr = data .- avgs'

        # Standardized moments: divide by the per-signal standard deviation
        if standardize
            cntr ./= _std(data, avgs, num_samples, num_signals, bias)'
        end
    end

    # Flattened representation (i.e., unfolded to a matrix)
    if flatten
        tensor = (Real)[]
        @inbounds for i = 1:num_signals
            face = zeros(num_signals, num_signals)
            @simd for j = 1:num_signals
                @simd for k = 1:num_signals
                    @inbounds face[j,k] = sum(cntr[:,i].*cntr[:,j].*cntr[:,k]) / adj
                end
            end
            @inbounds tensor = (i == 1) ? face : [tensor face]
        end

    # Cube: third-order tensor representation
    else
        tensor = zeros(num_signals, num_signals, num_signals)

        # Dense: all values are calculated, including duplicates
        if dense
            @simd for i = 1:num_signals
                @simd for j = 1:num_signals
                    @simd for k = 1:num_signals
                        @inbounds tensor[i,j,k] = sum(cntr[:,i].*cntr[:,j].*cntr[:,k]) / adj
                    end
                end
            end

        # Triangular: duplicate values ignored
        else
            @simd for i = 1:num_signals
                @simd for j = 1:i
                    @simd for k = 1:j
                        @inbounds tensor[i,j,k] = sum(cntr[:,i].*cntr[:,j].*cntr[:,k]) / adj
                    end
                end
            end
        end
    end
    return tensor
end

# Cokurtosis tensor (fourth-order)
function cokurt{T<:Real}(data::Matrix{T};
                         standardize::Bool=false,
                         flatten::Bool=false,
                         bias::Int=0,
                         dense::Bool=true)
    num_samples, num_signals = size(data)
    adj = num_samples - bias
    @inbounds begin
        avgs = vec(mean(data, 1))
        cntr = data .- avgs'

        # Standardized moments: divide by the per-signal standard deviation
        if standardize
            cntr ./= _std(data, avgs, num_samples, num_signals, bias)'
        end
    end

    # Flattened representation (i.e., unfolded into a matrix)
    if flatten
        tensor = (Real)[]
        @inbounds for i = 1:num_signals
            @inbounds for j = 1:num_signals
                face = zeros(num_signals, num_signals)
                @simd for k = 1:num_signals
                    @simd for l = 1:num_signals
                        @inbounds face[k,l] = sum(cntr[:,i].*cntr[:,j].*cntr[:,k].*cntr[:,l]) / adj
                    end
                end
                @inbounds tensor = (i == j == 1) ? face : [tensor face]
            end
        end

    # Tesseract: fourth-order tensor representation
    else
        tensor = zeros(num_signals, num_signals, num_signals, num_signals)

        # Dense: all values are calculated, including duplicates
        if dense
            @simd for i = 1:num_signals
                @simd for j = 1:num_signals
                    @simd for k = 1:num_signals
                        @simd for l = 1:num_signals
                            @inbounds tensor[i,j,k,l] = sum(cntr[:,i].*cntr[:,j].*cntr[:,k].*cntr[:,l]) / adj
                        end
                    end
                end
            end

        # Triangular: duplicate values ignored        
        else
            @simd for i = 1:num_signals
                @simd for j = 1:i
                    @simd for k = 1:j
                        @simd for l = 1:k
                            @inbounds tensor[i,j,k,l] = sum(cntr[:,i].*cntr[:,j].*cntr[:,k].*cntr[:,l]) / adj
                        end
                    end
                end
            end
        end
    end
    return tensor
end

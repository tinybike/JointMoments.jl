# Covariance matrix (for testing)
function _cov{T<:Real}(data::Matrix{T}; bias::Int=0, dense::Bool=true)
    num_samples, num_signals = size(data)
    adj = num_samples - bias
    @inbounds cntr = data .- mean(data, 1)

    # Dense matrix (all values are calculated, including duplicates)
    if dense
        tensor = zeros(num_signals, num_signals)
        @simd for i = 1:num_signals
            @simd for j = 1:num_signals
                @inbounds tensor[i,j] = sum(cntr[:,i].*cntr[:,j]) / adj
            end
        end

    # Lower triangular matrix (duplicate values not calculated)
    # Convert to dense matrix using:
    #   tensor + tensor' - diagm(diag(tensor)))
    else
        tensor = spzeros(num_signals, num_signals)
        @simd for i = 1:num_signals
            @simd for j = 1:i
                @inbounds tensor[i,j] = sum(cntr[:,i].*cntr[:,j]) / adj
            end
        end
    end
    tensor
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
            if bias == 1
                cntr ./= std(data, 1)
            else
                cntr ./= _std(data, avgs, num_samples, num_signals)'
            end
        end
    end

    # Flattened representation (i.e., unfolded to a matrix)
    if flatten

        # Dense: all values are calculated, including duplicates
        if dense
            tensor = zeros(num_signals, num_signals^2)
            @simd for i = 1:num_signals
                @simd for j = 1:num_signals
                    @simd for k = 1:num_signals
                        @inbounds tensor[j,(i-1)*num_signals+k] = sum(cntr[:,i].*cntr[:,j].*cntr[:,k]) / adj
                    end
                end
            end

        # Triangular: duplicate values ignored
        else
            tensor = spzeros(num_signals, num_signals^2)
            @simd for i = 1:num_signals
                @simd for j = 1:i
                    @simd for k = 1:j
                        @inbounds tensor[j,(i-1)*num_signals+k] = sum(cntr[:,i].*cntr[:,j].*cntr[:,k]) / adj
                    end
                end
            end
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
    tensor
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
            if bias == 1
                cntr ./= std(data, 1)
            else
                cntr ./= _std(data, avgs, num_samples, num_signals)'
            end
        end
    end

    # Flattened representation (i.e., unfolded into a matrix)
    if flatten

        # Dense: all values are calculated, including duplicates
        if dense
            tensor = zeros(num_signals, num_signals^3)
            @simd for i = 1:num_signals
                @simd for j = 1:num_signals
                    @simd for k = 1:num_signals
                        @simd for l = 1:num_signals
                            @inbounds tensor[k,(i-1)*num_signals^2 + (j-1)*num_signals + l] = sum(cntr[:,i].*cntr[:,j].*cntr[:,k].*cntr[:,l]) / adj
                        end
                    end
                end
            end

        # Triangular: duplicate values ignored
        else
            tensor = zeros(num_signals, num_signals^3)
            # tensor = spzeros(num_signals, num_signals^3)
            @simd for i = 1:num_signals
                @simd for j = 1:i
                    @simd for k = 1:j
                        @simd for l = 1:k
                            @inbounds tensor[k,(i-1)*num_signals^2 + (j-1)*num_signals + l] = sum(cntr[:,i].*cntr[:,j].*cntr[:,k].*cntr[:,l]) / adj
                        end
                    end
                end
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
    tensor
end

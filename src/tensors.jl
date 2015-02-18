function _stddev{T<:Real}(v::Array{T,1}, avg::Real, bias::Int)
    if bias == 0
        n = length(v)
        s = 0.0
        for i = 1:n
            @inbounds z = v[i] - avg
            s += z * z
        end
        sqrt(s / n)
    elseif bias == 1
        std(v)
    end
end

# Covariance matrix (for testing)
function _covar{T<:Real}(data::Matrix{T}; bias::Int=0, dense::Bool=true)
    num_samples, num_signals = size(data)
    tensor = zeros(num_signals, num_signals)

    # Averages along each dimension
    @inbounds begin
        i_mean = mean(data, 1)
        j_mean = mean(data, 2)
    end

    # Dense matrix (all values are calculated, including duplicates)
    if dense
        @simd for i = 1:num_signals
            @simd for j = 1:num_signals
                prod = 0
                @simd for t = 1:num_samples
                    @inbounds begin
                        i_term = data[t,i] - i_mean[i]
                        j_term = data[t,j] - j_mean[j]
                    end
                    prod +=  i_term * j_term
                end
                @inbounds tensor[i,j] = prod / (num_samples - bias)
            end
        end

    # Lower triangular matrix (duplicate values not calculated)
    # Convert to dense matrix using:
    #   tensor + tensor' - diagm(diag(tensor)))
    else
        @simd for i = 1:num_signals
            @simd for j = 1:i
                prod = 0
                @simd for t = 1:num_samples 
                    @inbounds begin
                        i_term = data[t,i] - i_mean[i]
                        j_term = data[t,j] - j_mean[j]
                    end
                    prod +=  i_term * j_term
                end
                @inbounds tensor[i,j] = prod / (num_samples - bias)
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

    # Averages along each dimension
    @inbounds begin
        i_mean = mean(data, 1)
        j_mean = mean(data, 2)
        k_mean = mean(data, 3)
    end

    # Flattened representation (i.e., unfolded into a matrix)
    if flatten
        tensor = []
        if standardize
            @inbounds for i = 1:num_signals
                i_std = _stddev(data[:,i], i_mean, bias)
                face = zeros(num_signals, num_signals)
                @inbounds for j = 1:num_signals
                    j_std = _stddev(data[:,j], j_mean, bias)
                    @inbounds for k = 1:num_signals
                        k_std = _stddev(data[:,k], k_mean, bias)
                        prod = 0
                        @inbounds for t = 1:num_samples 
                            i_term = (data[t,i] - i_mean[i]) / i_std
                            j_term = (data[t,j] - j_mean[j]) / j_std 
                            k_term = (data[t,k] - k_mean[k]) / k_std
                            prod += i_term * j_term * k_term
                        end
                        face[j,k] = prod / (num_samples - bias)
                    end
                end
                tensor = (i == 1) ? face : [tensor face]
            end
        else
            @inbounds for i = 1:num_signals
                face = zeros(num_signals, num_signals)
                @inbounds for j = 1:num_signals
                    @inbounds for k = 1:num_signals
                        prod = 0
                        @inbounds for t = 1:num_samples 
                            i_term = data[t,i] - i_mean[i]
                            j_term = data[t,j] - j_mean[j]
                            k_term = data[t,k] - k_mean[k]
                            prod += i_term * j_term * k_term
                        end
                        face[j,k] = prod / (num_samples - bias)
                    end
                end
                tensor = (i == 1) ? face : [tensor face]
            end
        end

    # Cube: third-order tensor representation
    else

        # Dense representation: all values are calculated, including duplicates
        if dense
            tensor = zeros(num_signals, num_signals, num_signals)

            # Standardized moments: divide each element by the standard
            # deviation of its fiber (along the active dimension)
            if standardize
                @simd for i = 1:num_signals
                    @inbounds i_std = _stddev(data[:,i], i_mean[i], bias)
                    @simd for j = 1:num_signals
                        @inbounds j_std = _stddev(data[:,j], j_mean[j], bias)
                        @simd for k = 1:num_signals
                            @inbounds k_std = _stddev(data[:,k], k_mean[k], bias)
                            prod = 0
                            @simd for t = 1:num_samples 
                                @inbounds begin
                                    i_term = (data[t,i] - i_mean[i]) / i_std
                                    j_term = (data[t,j] - j_mean[j]) / j_std 
                                    k_term = (data[t,k] - k_mean[k]) / k_std
                                end
                                prod += i_term * j_term * k_term
                            end
                            @inbounds tensor[i,j,k] = prod / (num_samples - bias)
                        end
                    end
                end

            # Unstandardized (raw) moments
            else
                @simd for i = 1:num_signals
                    @simd for j = 1:num_signals
                        @simd for k = 1:num_signals
                            prod = 0
                            @simd for t = 1:num_samples
                                @inbounds begin
                                    i_term = data[t,i] - i_mean[i]
                                    j_term = data[t,j] - j_mean[j]
                                    k_term = data[t,k] - k_mean[k]
                                end
                                prod += i_term * j_term * k_term
                            end
                            @inbounds tensor[i,j,k] = prod / (num_samples - bias)
                        end
                    end
                end
            end

        # Triangular representation: duplicate values ignored
        else
            tensor = zeros(num_signals, num_signals, num_signals)

            # Standardized moments: divide each element by the standard
            # deviation of its fiber (along the active dimension)
            if standardize
                @simd for i = 1:num_signals
                    @inbounds i_std = _stddev(data[:,i], i_mean[i], bias)
                    @simd for j = 1:i
                        @inbounds j_std = _stddev(data[:,j], j_mean[j], bias)
                        @simd for k = 1:j
                            @inbounds k_std = _stddev(data[:,k], k_mean[k], bias)
                            prod = 0
                            @simd for t = 1:num_samples
                                @inbounds begin
                                    i_term = (data[t,i] - i_mean[i]) / i_std
                                    j_term = (data[t,j] - j_mean[j]) / j_std 
                                    k_term = (data[t,k] - k_mean[k]) / k_std
                                end
                                prod += i_term * j_term * k_term
                            end
                            @inbounds tensor[i,j,k] = prod / (num_samples - bias)
                        end
                    end
                end

            # Unstandardized (raw) moments
            else
                @simd for i = 1:num_signals
                    @simd for j = 1:i
                        @simd for k = 1:j
                            prod = 0
                            @simd for t = 1:num_samples
                                @inbounds begin
                                    i_term = data[t,i] - i_mean[i]
                                    j_term = data[t,j] - j_mean[j]
                                    k_term = data[t,k] - k_mean[k]
                                end
                                prod += i_term * j_term * k_term
                            end
                            @inbounds tensor[i,j,k] = prod / (num_samples - bias)
                        end
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

    # Averages along each dimension
    @inbounds begin
        i_mean = mean(data, 1)
        j_mean = mean(data, 2)
        k_mean = mean(data, 3)
        l_mean = mean(data, 4)
    end

    # Flattened representation (i.e., unfolded into a matrix)
    if flatten
        tensor = []
        if standardize
            @inbounds for i = 1:num_signals
                i_std = _stddev(data[:,i], i_mean[i], bias)
                @inbounds for j = 1:num_signals
                    j_std = _stddev(data[:,j], j_mean[j], bias)
                    face = zeros(num_signals, num_signals)
                    @inbounds for k = 1:num_signals
                        k_std = _stddev(data[:,k], k_mean[k], bias)
                        @inbounds for l = 1:num_signals
                            l_std = _stddev(data[:,l], l_mean[l], bias)
                            prod = 0
                            @inbounds for t = 1:num_samples 
                                i_term = (data[t,i] - i_mean[i]) / i_std
                                j_term = (data[t,j] - j_mean[j]) / j_std 
                                k_term = (data[t,k] - k_mean[k]) / k_std
                                l_term = (data[t,l] - l_mean[l]) / l_std
                                prod += i_term * j_term * k_term * l_term
                            end
                            face[k,l] = prod / (num_samples - bias)
                        end
                    end
                    tensor = (i == j == 1) ? face : [tensor face]
                end
            end
        else
            @inbounds for i = 1:num_signals
                @inbounds for j = 1:num_signals
                    face = zeros(num_signals, num_signals)
                    @inbounds for k = 1:num_signals
                        @inbounds for l = 1:num_signals
                            prod = 0
                            @inbounds for t = 1:num_samples
                                i_term = data[t,i] - i_mean[i]
                                j_term = data[t,j] - j_mean[j]
                                k_term = data[t,k] - k_mean[k]
                                l_term = data[t,l] - l_mean[l]
                                prod += i_term * j_term * k_term * l_term
                            end
                            face[k,l] = prod / (num_samples - bias)
                        end
                    end
                    tensor = (i == j == 1) ? face : [tensor face]
                end
            end
        end

    # Tesseract: fourth-order tensor representation
    else
        tensor = zeros(num_signals, num_signals, num_signals, num_signals)

        # Dense representation: all values are calculated, including duplicates
        if dense

            # Standardized moments: divide each element by the standard
            # deviation of its fiber (along the active dimension)
            if standardize
                @simd for i = 1:num_signals
                    @inbounds i_std = _stddev(data[:,i], i_mean[i], bias)
                    @simd for j = 1:num_signals
                        @inbounds j_std = _stddev(data[:,j], j_mean[j], bias)
                        @simd for k = 1:num_signals
                            @inbounds k_std = _stddev(data[:,k], k_mean[k], bias)
                            @simd for l = 1:num_signals
                                @inbounds l_std = _stddev(data[:,l], l_mean[l], bias)
                                prod = 0
                                @simd for t = 1:num_samples
                                    @inbounds begin
                                        i_term = (data[t,i] - i_mean[i]) / i_std
                                        j_term = (data[t,j] - j_mean[j]) / j_std 
                                        k_term = (data[t,k] - k_mean[k]) / k_std
                                        l_term = (data[t,l] - l_mean[l]) / l_std
                                    end
                                    prod += i_term * j_term * k_term * l_term
                                end
                                @inbounds tensor[i,j,k,l] = prod / (num_samples - bias)
                            end
                        end
                    end
                end

            # Unstandardized (raw) moments
            else
                @simd for i = 1:num_signals
                    @simd for j = 1:num_signals
                        @simd for k = 1:num_signals
                            @simd for l = 1:num_signals
                                prod = 0
                                @simd for t = 1:num_samples
                                    @inbounds begin
                                        i_term = data[t,i] - i_mean[i]
                                        j_term = data[t,j] - j_mean[j]
                                        k_term = data[t,k] - k_mean[k]
                                        l_term = data[t,l] - l_mean[l]
                                    end
                                    prod += i_term * j_term * k_term * l_term
                                end
                                @inbounds tensor[i,j,k,l] = prod / (num_samples - bias)
                            end
                        end
                    end
                end
            end

        # Triangular representation: duplicate values ignored        
        else
            # Standardized moments: divide each element by the standard
            # deviation of its fiber (along the active dimension)
            if standardize
                @simd for i = 1:num_signals
                    @inbounds i_std = _stddev(data[:,i], i_mean[i], bias)
                    @simd for j = 1:i
                        @inbounds j_std = _stddev(data[:,j], j_mean[j], bias)
                        @simd for k = 1:j
                            @inbounds k_std = _stddev(data[:,k], k_mean[k], bias)
                            @simd for l = 1:k
                                @inbounds l_std = _stddev(data[:,l], l_mean[l], bias)
                                prod = 0
                                @simd for t = 1:num_samples
                                    @inbounds begin
                                        i_term = (data[t,i] - i_mean[i]) / i_std
                                        j_term = (data[t,j] - j_mean[j]) / j_std 
                                        k_term = (data[t,k] - k_mean[k]) / k_std
                                        l_term = (data[t,l] - l_mean[l]) / l_std
                                    end
                                    prod += i_term * j_term * k_term * l_term
                                end
                                @inbounds tensor[i,j,k,l] = prod / (num_samples - bias)
                            end
                        end
                    end
                end

            # Unstandardized (raw) moments
            else
                @simd for i = 1:num_signals
                    @simd for j = 1:i
                        @simd for k = 1:j
                            @simd for l = 1:k
                                prod = 0
                                @simd for t = 1:num_samples
                                    @inbounds begin
                                        i_term = data[t,i] - i_mean[i]
                                        j_term = data[t,j] - j_mean[j]
                                        k_term = data[t,k] - k_mean[k]
                                        l_term = data[t,l] - l_mean[l]
                                    end
                                    prod += i_term * j_term * k_term * l_term
                                end
                                @inbounds tensor[i,j,k,l] = prod / (num_samples - bias)
                            end
                        end
                    end
                end
            end
        end
    end
    return tensor
end

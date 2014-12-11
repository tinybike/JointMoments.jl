function _stddev{T<:Real}(v::Array{T,1}, m::Real, bias)
    if bias == 0
        n = length(v)
        s = 0.0
        for i = 1:n
            @inbounds z = v[i] - m
            s += z * z
        end
        sqrt(s / n)
    elseif bias == 1
        std(v)
    end
end

function coskew{T<:Real}(data::Array{T}; standardize=false, flatten=false, bias=0)
    num_samples, num_signals = size(data)
    if flatten
        tensor = []
        if standardize
            @inbounds for i = 1:num_signals
                i_mean = mean(data[:,i])
                i_std = _stddev(data[:,i], i_mean, bias)
                face = zeros(num_signals, num_signals)
                @inbounds for j = 1:num_signals
                    j_mean = mean(data[:,j])
                    j_std = _stddev(data[:,j], j_mean, bias)
                    @inbounds for k = 1:num_signals
                        k_mean = mean(data[:,k])
                        k_std = _stddev(data[:,k], k_mean, bias)
                        prod = 0
                        @inbounds for t = 1:num_samples 
                            i_term = (data[t,i] - i_mean) / i_std
                            j_term = (data[t,j] - j_mean) / j_std 
                            k_term = (data[t,k] - k_mean) / k_std
                            prod +=  i_term * j_term * k_term
                        end
                        face[j,k] = prod / (num_samples - bias)
                    end
                end
                tensor = (i == 1) ? face : [tensor face]
            end
        else
            @inbounds for i = 1:num_signals
                i_mean = mean(data[:,i])
                face = zeros(num_signals, num_signals)
                @inbounds for j = 1:num_signals
                    j_mean = mean(data[:,j])
                    @inbounds for k = 1:num_signals
                        k_mean = mean(data[:,k])
                        prod = 0
                        @inbounds for t = 1:num_samples 
                            i_term = data[t,i] - i_mean
                            j_term = data[t,j] - j_mean
                            k_term = data[t,k] - k_mean
                            prod +=  i_term * j_term * k_term
                        end
                        face[j,k] = prod / (num_samples - bias)
                    end
                end
                tensor = (i == 1) ? face : [tensor face]
            end
        end
    else
        tensor = zeros(num_signals, num_signals, num_signals)
        if standardize
            @inbounds for i = 1:num_signals
                i_mean = mean(data[:,i])
                i_std = _stddev(data[:,i], i_mean, bias)
                @inbounds for j = 1:num_signals
                    j_mean = mean(data[:,j])
                    j_std = _stddev(data[:,j], j_mean, bias)
                    @inbounds for k = 1:num_signals
                        k_mean = mean(data[:,k])
                        k_std = _stddev(data[:,k], k_mean, bias)
                        prod = 0
                        @inbounds for t = 1:num_samples 
                            i_term = (data[t,i] - i_mean) / i_std
                            j_term = (data[t,j] - j_mean) / j_std 
                            k_term = (data[t,k] - k_mean) / k_std
                            prod +=  i_term * j_term * k_term
                        end
                        tensor[i,j,k] = prod / (num_samples - bias)
                    end
                end
            end
        else
            @inbounds for i = 1:num_signals
                i_mean = mean(data[:,i])
                @inbounds for j = 1:num_signals
                    j_mean = mean(data[:,j])
                    @inbounds for k = 1:num_signals
                        k_mean = mean(data[:,k])
                        prod = 0
                        @inbounds for t = 1:num_samples 
                            i_term = data[t,i] - i_mean
                            j_term = data[t,j] - j_mean
                            k_term = data[t,k] - k_mean
                            prod +=  i_term * j_term * k_term
                        end
                        tensor[i,j,k] = prod / (num_samples - bias)
                    end
                end
            end
        end
    end
    return tensor
end

function cokurt{T<:Real}(data::Array{T}; standardize=false, flatten=false, bias=0)
    num_samples, num_signals = size(data)
    if flatten
        tensor = []
        if standardize
            @inbounds for i = 1:num_signals
                i_mean = mean(data[:,i])
                i_std = _stddev(data[:,i], i_mean, bias)
                @inbounds for j = 1:num_signals
                    j_mean = mean(data[:,j])
                    j_std = _stddev(data[:,j], j_mean, bias)
                    face = zeros(num_signals, num_signals)
                    @inbounds for k = 1:num_signals
                        k_mean = mean(data[:,k])
                        k_std = _stddev(data[:,k], k_mean, bias)
                        @inbounds for l = 1:num_signals
                            l_mean = mean(data[:,l])
                            l_std = _stddev(data[:,l], l_mean, bias)
                            prod = 0
                            @inbounds for t = 1:num_samples 
                                i_term = (data[t,i] - i_mean) / i_std
                                j_term = (data[t,j] - j_mean) / j_std 
                                k_term = (data[t,k] - k_mean) / k_std
                                l_term = (data[t,l] - l_mean) / l_std
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
                i_mean = mean(data[:,i])
                @inbounds for j = 1:num_signals
                    j_mean = mean(data[:,j])
                    face = zeros(num_signals, num_signals)
                    @inbounds for k = 1:num_signals
                        k_mean = mean(data[:,k])
                        @inbounds for l = 1:num_signals
                            l_mean = mean(data[:,l])
                            prod = 0
                            @inbounds for t = 1:num_samples 
                                i_term = data[t,i] - i_mean
                                j_term = data[t,j] - j_mean 
                                k_term = data[t,k] - k_mean
                                l_term = data[t,l] - l_mean
                                prod += i_term * j_term * k_term * l_term
                            end
                            face[k,l] = prod / (num_samples - bias)
                        end
                    end
                    tensor = (i == j == 1) ? face : [tensor face]
                end
            end
        end
    else
        tensor = zeros(num_signals, num_signals, num_signals, num_signals)
        if standardize
            @inbounds for i = 1:num_signals
                i_mean = mean(data[:,i])
                i_std = _stddev(data[:,i], i_mean, bias)
                @inbounds for j = 1:num_signals
                    j_mean = mean(data[:,j])
                    j_std = _stddev(data[:,j], j_mean, bias)
                    @inbounds for k = 1:num_signals
                        k_mean = mean(data[:,k])
                        k_std = _stddev(data[:,k], k_mean, bias)
                        @inbounds for l = 1:num_signals
                            l_mean = mean(data[:,l])
                            l_std = _stddev(data[:,l], l_mean, bias)
                            prod = 0
                            @inbounds for t = 1:num_samples 
                                i_term = (data[t,i] - i_mean) / i_std
                                j_term = (data[t,j] - j_mean) / j_std 
                                k_term = (data[t,k] - k_mean) / k_std
                                l_term = (data[t,l] - l_mean) / l_std
                                prod += i_term * j_term * k_term * l_term
                            end
                            tensor[i,j,k,l] = prod / (num_samples - bias)
                        end
                    end
                end
            end
        else
            @inbounds for i = 1:num_signals
                i_mean = mean(data[:,i])
                @inbounds for j = 1:num_signals
                    j_mean = mean(data[:,j])
                    @inbounds for k = 1:num_signals
                        k_mean = mean(data[:,k])
                        @inbounds for l = 1:num_signals
                            l_mean = mean(data[:,l])
                            prod = 0
                            @inbounds for t = 1:num_samples 
                                i_term = data[t,i] - i_mean
                                j_term = data[t,j] - j_mean
                                k_term = data[t,k] - k_mean
                                l_term = data[t,l] - l_mean
                                prod += i_term * j_term * k_term * l_term
                            end
                            tensor[i,j,k,l] = prod / (num_samples - bias)
                        end
                    end
                end
            end
        end
    end
    return tensor
end

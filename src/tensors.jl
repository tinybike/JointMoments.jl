# Outer (tensor) product
function outer{N,a}(A::Array{N,a}, B::Vector{N})
    reshape(N[x*y for x in A, y in B], size(A)..., size(B)...)::Array{N,a+1}
end

# Covariance matrix with adjustable bias (Bessel's correction)
function _cov{T<:Real}(data::Matrix{T}; bias::Int=0, dense::Bool=true)
    num_samples, num_signals = size(data)
    @inbounds cntr = data .- mean(data, 1)

    # Dense matrix: all values are calculated, including duplicates
    if dense
        tensor = zeros(num_signals, num_signals)
        @inbounds for i = 1:num_samples
            c = cntr[i,:]
            tensor += c' * c
        end

    # Lower triangular matrix (duplicate values not calculated)
    # Convert to dense matrix using:
    #   tensor + tensor' - diagm(diag(tensor)))
    else
        tensor = zeros(num_signals, num_signals)
        @simd for i = 1:num_signals
            @simd for j = 1:i
                @inbounds tensor[i,j] = sum(cntr[:,i].*cntr[:,j])
            end
        end
    end
    tensor / (num_samples - bias)
end

# Coskewness tensor (third-order)
function coskew{T<:Real}(data::Matrix{T};
                         standardize::Bool=false,
                         flatten::Bool=false,
                         bias::Int=0,
                         dense::Bool=true)

    # Center/whiten data
    cntr, num_samples, num_signals = center(data;
                                            standardize=standardize,
                                            bias=bias)

    # Flattened representation (i.e., unfolded to a matrix)
    if flatten

        # Dense: all values are calculated, including duplicates
        if dense
            tensor = zeros(num_signals, num_signals^2)
            @inbounds for i = 1:num_samples
                c = cntr[i,:]
                tensor += kron(c'*c, c)
            end

        # Triangular: duplicate values ignored
        else
            tensor = spzeros(num_signals, num_signals^2)
            @simd for i = 1:num_signals
                @simd for j = 1:i
                    @simd for k = 1:j
                        @inbounds tensor[j,(i-1)*num_signals+k] = sum(cntr[:,i].*cntr[:,j].*cntr[:,k])
                    end
                end
            end
        end

    # Cube: third-order tensor representation
    else
        tensor = zeros(num_signals, num_signals, num_signals)

        # Dense: all values are calculated, including duplicates
        if dense
            @inbounds for i = 1:num_samples
                c = cntr[i,:][:]
                tensor += outer(c*c', c)
            end

        # Triangular: duplicate values ignored
        else
            @simd for i = 1:num_signals
                @simd for j = 1:i
                    @simd for k = 1:j
                        @inbounds tensor[i,j,k] = sum(cntr[:,i].*cntr[:,j].*cntr[:,k])
                    end
                end
            end
        end
    end
    tensor / (num_samples - bias)
end

# Cokurtosis tensor (fourth-order)
function cokurt{T<:Real}(data::Matrix{T};
                         standardize::Bool=false,
                         flatten::Bool=false,
                         bias::Int=0,
                         dense::Bool=true)

    # Center/whiten data
    cntr, num_samples, num_signals = center(data;
                                            standardize=standardize,
                                            bias=bias)

    # Flattened representation (i.e., unfolded into a matrix)
    if flatten

        # Dense: all values are calculated, including duplicates
        if dense
            tensor = zeros(num_signals, num_signals^3)
            @inbounds for i = 1:num_samples
                c = cntr[i,:]
                tensor += kron(c'*c, c, c)
            end

        # Triangular: duplicate values ignored
        else
            tensor = spzeros(num_signals, num_signals^3)
            @simd for i = 1:num_signals
                @simd for j = 1:i
                    @simd for k = 1:j
                        @simd for l = 1:k
                            @inbounds tensor[k,(i-1)*num_signals^2 + (j-1)*num_signals + l] = sum(cntr[:,i].*cntr[:,j].*cntr[:,k].*cntr[:,l])
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
            @inbounds for i = 1:num_samples
                c = cntr[i,:][:]
                tensor += outer(outer(c*c', c), c)
            end

        # Triangular: duplicate values ignored
        else
            @simd for i = 1:num_signals
                @simd for j = 1:i
                    @simd for k = 1:j
                        @simd for l = 1:k
                            @inbounds tensor[i,j,k,l] = sum(cntr[:,i].*cntr[:,j].*cntr[:,k].*cntr[:,l])
                        end
                    end
                end
            end
        end
    end
    tensor / (num_samples - bias)
end

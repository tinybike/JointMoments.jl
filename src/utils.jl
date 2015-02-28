# Swap dimensions of a tensor (ndarray-transpose)
function _transpose{T<:Real}(tensor::Array{T,3}, permute::Vector{Int})
    shape = size(tensor)
    ttensor = zeros(shape)
    idx = zeros(3)
    for idx[1] = 1:shape[1]
        for idx[2] = 1:shape[2]
            for idx[3] = 1:shape[3]
                ttensor[idx[permute]...] = tensor[idx...]
            end
        end
    end
    ttensor
end

# Swap dimensions of a tensor (ndarray-transpose)
function _transpose{T<:Real}(tensor::Array{T,4}, permute::Vector{Int})
    shape = size(tensor)
    ttensor = zeros(shape)
    idx = zeros(4)
    for idx[1] = 1:shape[1]
        for idx[2] = 1:shape[2]
            for idx[3] = 1:shape[3]
                for idx[4] = 1:shape[4]
                    ttensor[idx[permute]...] = tensor[idx...]
                end
            end
        end
    end
    ttensor
end

function _corners{T<:Real}(tensor::Array{T})
    shape = size(tensor)
    N = ndims(tensor)
    z = zeros(N)
    corners = zeros(shape...)
    for i = 1:shape[1]
        idx = z + i
        corners[idx...] = tensor[idx...]
    end
    corners
end

function _pairs{T<:Real}(tensor::Array{T,3}, swapped::Vector{Int})
    shape = size(tensor)
    N = ndims(tensor)
    pairs = zeros(shape)
    swapped = sort(swapped)
    for i = 1:shape[1]
        if swapped == [1,2]
            pairs[:,:,i] = _corners(convert(Array, slice(tensor, :, :, i)))
        elseif swapped == [1,3]
            pairs[:,i,:] = _corners(convert(Array, slice(tensor, :, i, :)))
        elseif swapped == [2,3]
            pairs[i,:,:] = _corners(convert(Array, slice(tensor, i, :, :)))
        end
    end
    pairs
end

function _dense{T<:Real}(tensor::Array{T})
    refilled = similar(tensor)
    permute = [1:ndims(tensor)]
    for p in permutations(permute)
        refilled += _transpose(tensor, p) - _pairs(tensor, find(p.!=permute))
    end
    refilled
end

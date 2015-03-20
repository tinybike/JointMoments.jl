tolerance(expected, observed) = maxpcterr(expected, observed) < ε

maxpcterr(expected::Array, observed::Array) = maximum(vec(pcterr(expected, observed)))

maxpcterr{T<:Number}(expected::T, observed::T) = maximum(pcterr(expected, observed))

pcterr(expected, observed) = abs(expected - observed) ./ expected

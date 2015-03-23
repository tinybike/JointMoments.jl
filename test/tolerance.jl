tolerance(expected, observed) = maxpcterr(expected, observed) < ε

maxpcterr(expected::Array, observed::Array) = maximum(vec(pcterr(expected, observed)))

maxpcterr{T<:Number}(expected::T, observed::T) = maximum(pcterr(expected, observed))

pcterr(expected, observed) = abs(expected - observed) ./ expected

function hcaulk(a, b)
    z = zeros(size(a, 1))
    display(round([a z b], 4))
    println("")
end

function vcaulk(a, b)
    z = zeros(size(a, 2))'
    display(round([a; z; b], 4))
    println("")
end

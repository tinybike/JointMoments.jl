# JointMoments

[![Build Status](https://travis-ci.org/tensorjack/JointMoments.jl.svg?branch=master)](https://travis-ci.org/tensorjack/JointMoments.jl) [![Coverage Status](https://coveralls.io/repos/tensorjack/JointMoments.jl/badge.png)](https://coveralls.io/r/tensorjack/JointMoments.jl)

Tensors and statistics for joint central moments.

### Usage

Installation:

    julia> Pkg.add("JointMoments")

Use:

    julia> using JointMoments

    julia> data = [ 0.837698   0.49452   2.54352 
                   -0.294096  -0.39636   0.728619
                   -1.62089   -0.44919   1.20592 
                   -1.06458   -0.68214  -1.12841 ];

    julia> weights = [1.0, 0.1, 0.5];

    julia> coskew(data)
    3x3x3 Array{Float64,3}:
    [:, :, 1] =
     0.294091  0.26697   0.773618
     0.26697   0.162051  0.350696
     0.773618  0.350696  0.451934

    [:, :, 2] =
     0.26697   0.162051   0.350696
     0.162051  0.0852269  0.156275
     0.350696  0.156275   0.131448

    [:, :, 3] =
     0.773618  0.350696   0.451934
     0.350696  0.156275   0.131448
     0.451934  0.131448  -0.645484

    julia> coskewness(data, weights)
    0.467586

    julia> cokurt(data)
    3x3x3x3 Array{Float64,4}:
    [:, :, 1, 1] =
     1.2563    0.563538  1.05898 
     0.563538  0.290737  0.643266
     1.05898   0.643266  1.68278 

    [:, :, 2, 1] =
     0.563538  0.290737  0.643266
     0.290737  0.158262  0.374873
     0.643266  0.374873  0.97585 

    [:, :, 3, 1] =
     1.05898   0.643266  1.68278
     0.643266  0.374873  0.97585
     1.68278   0.97585   2.69607

    [:, :, 1, 2] =
     0.563538  0.290737  0.643266
     0.290737  0.158262  0.374873
     0.643266  0.374873  0.97585 

    [:, :, 2, 2] =
     0.290737  0.158262   0.374873
     0.158262  0.0887859  0.218824
     0.374873  0.218824   0.58726 

    [:, :, 3, 2] =
     0.643266  0.374873  0.97585
     0.374873  0.218824  0.58726
     0.97585   0.58726   1.73728

    [:, :, 1, 3] =
     1.05898   0.643266  1.68278
     0.643266  0.374873  0.97585
     1.68278   0.97585   2.69607

    [:, :, 2, 3] =
     0.643266  0.374873  0.97585
     0.374873  0.218824  0.58726
     0.97585   0.58726   1.73728

    [:, :, 3, 3] =
     1.68278  0.97585  2.69607
     0.97585  0.58726  1.73728
     2.69607  1.73728  5.85635

    julia> cokurtosis(data, weights)
    1-element Array{Float64,1}:
     1.32039

The `coskew` and `cokurt` functions can also return flattened/unfolded versions of the cumulant tensors:

    julia> coskew(data, flatten=true)
    3x9 Array{Float64,2}:
     0.294091  0.26697   0.773618  0.26697   0.162051   0.350696  0.773618  0.350696   0.451934
     0.26697   0.162051  0.350696  0.162051  0.0852269  0.156275  0.350696  0.156275   0.131448
     0.773618  0.350696  0.451934  0.350696  0.156275   0.131448  0.451934  0.131448  -0.645484


### Tests

    $ julia test/runtests.jl

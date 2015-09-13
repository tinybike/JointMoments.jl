# JointMoments

[![Build Status](https://travis-ci.org/tinybike/JointMoments.jl.svg?branch=master)](https://travis-ci.org/tinybike/JointMoments.jl) [![Coverage Status](https://coveralls.io/repos/tinybike/JointMoments.jl/badge.svg?branch=master)](https://coveralls.io/r/tinybike/JointMoments.jl?branch=master) [![JointMoments](http://pkg.julialang.org/badges/JointMoments_0.3.svg)](http://pkg.julialang.org/?pkg=JointMoments&ver=0.3)

Tensors and statistics for third and fourth joint central moments.

### Usage

Installation:

    julia> Pkg.add("JointMoments")

Use:

    julia> using JointMoments

    julia> data = [ 0.837698   0.49452   2.54352 
                   -0.294096  -0.39636   0.728619
                   -1.62089   -0.44919   1.20592 
                   -1.06458   -0.68214  -1.12841 ];

Third and fourth joint central moment tensors:

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

Statistics:

    julia> coskewness(data)
    0.2838850631006579

    julia> cokurtosis(data)
    0.8916763961210045

`coskewness` and `cokurtosis` can use an optional weight vector, which assigns a weight to each column of the data matrix:

    julia> weights = [1.0, 0.1, 0.5];

    julia> coskewness(data, weights)
    0.46758589701357833

    julia> cokurtosis(data, weights)
    1.3203902349727108

The `coskew` and `cokurt` functions can also return flattened/unfolded tensors:

    julia> coskew(data, flatten=true)
    3x9 Array{Float64,2}:
     0.294091  0.26697   0.773618  0.26697   0.162051   0.350696  0.773618  0.350696   0.451934
     0.26697   0.162051  0.350696  0.162051  0.0852269  0.156275  0.350696  0.156275   0.131448
     0.773618  0.350696  0.451934  0.350696  0.156275   0.131448  0.451934  0.131448  -0.645484

     julia> cokurt(data,flatten=true)
    3x27 Array{Float64,2}:
     2.12678   1.11885   0.474782  1.11885    1.12294     0.187331   0.474782   0.187331   1.15524   1.11885   …  0.276558  0.474782   0.187331   1.15524    0.187331   -0.0266349  0.276558  1.15524   0.276558  0.178083
     1.11885   1.12294   0.187331  1.12294    1.40462    -0.0266349  0.187331  -0.0266349  0.276558  1.12294      0.779221  0.187331  -0.0266349  0.276558  -0.0266349  -0.517198   0.779221  0.276558  0.779221  0.218732
     0.474782  0.187331  1.15524   0.187331  -0.0266349   0.276558   1.15524    0.276558   0.178083  0.187331     0.218732  1.15524    0.276558   0.178083   0.276558    0.779221   0.218732  0.178083  0.218732  5.98947 

The `coskew`, `cokurt`, `coskewness`, and `cokurtosis` functions have `standardize` and `bias` keyword arguments.  Setting `standardize=true` standardizes the elements of the joint moment tensors (divides by the standard deviation).  Setting `bias=1` uses Bessel's correction (divides by `N-1` instead of `N`).

### Tests

Unit tests can be run from the command line:

    $ julia test/runtests.jl

Or from the Julia prompt:

    julia> Pkg.test("JointMoments")

This package includes a rudimentary timing framework in `test/timing.jl`.  To run the timed examples:

    $ julia --color test/timing.jl

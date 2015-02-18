using JointMoments

include("data.jl")

time_covar = true
time_coskew = true
time_cokurt = true

testdata = bigdata
# for testdata in (data, bigdata)
    rows, cols = size(testdata)
    label = string("Test data: ", rows, "x", cols, " ", typeof(testdata), "\n")
    print_with_color(:white, label)
    for i = 1:2
        print_with_color(:green, "Trial $i\n")
        if time_covar
            print_with_color(:cyan, "_covar (dense=true)  ")
            @time S = _covar(testdata, dense=true, bias=1)
            print_with_color(:cyan, "_covar (dense=false) ")
            @time S = _covar(testdata, dense=false, bias=1)
        end
        if time_coskew
            print_with_color(:cyan, "coskew (dense=true, standardize=true)   ")
            @time S = coskew(testdata, dense=true, standardize=true, bias=1)
            print_with_color(:cyan, "coskew (dense=true, standardize=false)  ")
            @time S = coskew(testdata, dense=true, standardize=false, bias=1)
            print_with_color(:cyan, "coskew (dense=false, standardize=true)  ")
            @time S = coskew(testdata, dense=false, standardize=true, bias=1)
            print_with_color(:cyan, "coskew (dense=false, standardize=false) ")
            @time S = coskew(testdata, dense=false, standardize=false, bias=1)
        end
        if time_cokurt
            print_with_color(:cyan, "cokurt (dense=true, standardize=true)   ")
            @time K = cokurt(testdata, dense=true, standardize=true, bias=1)
            print_with_color(:cyan, "cokurt (dense=true, standardize=false)  ")
            @time K = cokurt(testdata, dense=true, standardize=false, bias=1)
            print_with_color(:cyan, "cokurt (dense=false, standardize=true)  ")
            @time K = cokurt(testdata, dense=false, standardize=true, bias=1)
            print_with_color(:cyan, "cokurt (dense=false, standardize=false) ")
            @time K = cokurt(testdata, dense=false, standardize=false, bias=1)
        end
    end
# end

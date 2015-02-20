using DataFrames
using JointMoments

include("data.jl")

function timing()
    time_covar = false
    time_coskew = true
    time_cokurt = false
    # datasets = (data, bigdata)
    datasets = (bigdata,)
    for (idx, dataset) in enumerate(datasets)
        df = DataFrame()
        rows, cols = size(dataset)
        label = string("Data: ", rows, "x", cols, " ", typeof(dataset), "\n")
        print_with_color(:white, label)
        for i = 1:2
            row = Dict()
            params = [(true, true), (true, false), (false, true), (false, false)]
            for p in params
                row[:bias] = 1
                row[:flatten] = false
                row[:dense] = p[1]
                if time_covar
                    row[:standardize] = NaN
                    row[:function] = :_covar
                    row[:elapsed] = @elapsed _covar(
                        dataset,
                        dense=row[:dense],
                        bias=1,
                    )
                    if i == 2
                        if size(df) == (0,0)
                            for (key, value) in row
                                df[key] = value
                            end
                        else
                            push!(df, [value for (key, value) in row])
                        end
                    end
                end
                if time_coskew
                    row[:dense] = p[1]
                    row[:standardize] = p[2]
                    row[:function] = :coskew
                    row[:elapsed] = @elapsed coskew(
                        dataset,
                        standardize=row[:standardize],
                        flatten=row[:flatten],
                        bias=row[:bias],
                        dense=row[:dense],
                    )
                    if i == 2
                        if size(df) == (0,0)
                            for (key, value) in row
                                df[key] = value
                            end
                        else
                            push!(df, [value for (key, value) in row])
                        end
                    end
                end
                if time_cokurt
                    row[:dense] = p[1]
                    row[:standardize] = p[2]
                    row[:function] = :cokurt
                    row[:elapsed] = @elapsed cokurt(
                        dataset,
                        dense=row[:dense],
                        standardize=row[:standardize],
                        bias=1,
                    )
                    if i == 2
                        if size(df) == (0,0)
                            for (key, value) in row
                                df[key] = value
                            end
                        else
                            push!(df, [value for (key, value) in row])
                        end
                    end
                end
            end
        end
        df = sort(df, cols=:elapsed)
        println(df)
    end
end

timing()

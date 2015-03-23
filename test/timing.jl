using DataFrames
using JointMoments

include("data.jl")

const ITERMAX = 25

function timing()
    time_cov = true
    time_coskew = true
    time_cokurt = true
    datasets = (data, bigdata)
    # datasets = (bigdata,)
    for (idx, dataset) in enumerate(datasets)
        df = DataFrame()
        rows, cols = size(dataset)
        label = string("Data: ", rows, "x", cols, " ", typeof(dataset), "\n")
        print_with_color(:white, label)
        for i = 1:2
            itermax = (i == 1) ? 1 : ITERMAX
            row = {
                :bias => 1,
                :flatten => false,
                :standardize => true,
                :function => :_cov,
                :elapsed => 0.0,
                :elapsed_error => 0.0,
            }
            # parameters: dense, standardize, flatten
            params = [
                (true, true, true),
                (true, true, false),
                (true, false, true),
                (false, true, true),
                (true, false, false),
                (false, true, false),
                (false, false, true),
                (false, false, false),
            ]
            for p in params
                row[:dense] = p[1]
                if time_cov
                    row[:standardize] = NaN
                    row[:flatten] = false
                    row[:function] = :_cov
                    elapse = (Float64)[]
                    for j = 1:itermax
                        push!(elapse,
                            @elapsed _cov(
                                dataset,
                                dense=row[:dense],
                                bias=row[:bias],
                            )
                        )
                    end
                    row[:elapsed] = mean(elapse)
                    row[:elapsed_error] = std(elapse) / ITERMAX
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
                    row[:flatten] = p[3]
                    row[:function] = :coskew
                    elapse = (Float64)[]
                    for j = 1:itermax
                        push!(elapse,
                            @elapsed coskew(
                                dataset,
                                standardize=row[:standardize],
                                flatten=row[:flatten],
                                bias=row[:bias],
                                dense=row[:dense],
                            )
                        )
                    end
                    row[:elapsed] = mean(elapse)
                    row[:elapsed_error] = std(elapse) / ITERMAX
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
                    row[:flatten] = p[3]
                    row[:function] = :cokurt
                    elapse = (Float64)[]
                    for j = 1:itermax
                        push!(elapse,
                            @elapsed cokurt(
                                dataset,
                                standardize=row[:standardize],
                                flatten=row[:flatten],
                                bias=row[:bias],
                                dense=row[:dense],
                            )
                        )
                    end
                    row[:elapsed] = mean(elapse)
                    row[:elapsed_error] = std(elapse) / ITERMAX
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

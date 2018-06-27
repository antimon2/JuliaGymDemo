# piecewise_schedule.jl
# original: https://github.com/denizyuret/Knet.jl/blob/master/examples/reinforcement-learning/dqn/piecewise_schedule.jl

linear_interpolation(l, r, α) = l + α * (r - l)

struct PiecewiseSchedule
    endpoints::Vector{Tuple{Int, Float64}}
end

function value(schedule::PiecewiseSchedule, t::Int)
    for ((l_t, l), (r_t, r)) in zip(schedule.endpoints[1:end-1], schedule.endpoints[2:end])
        if l_t <= t < r_t
            α = (t - l_t) / (r_t - l_t)
            return linear_interpolation(l, r, α)
        end
    end
    return schedule.endpoints[end][2]
end
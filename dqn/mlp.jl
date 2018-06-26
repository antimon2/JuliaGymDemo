# mlp.jl
# original: https://github.com/denizyuret/Knet.jl/blob/master/examples/reinforcement-learning/dqn/mlp.jl

using AutoGrad

const ArrayOrKnetArray{T} = Union{<:Array{T}, <:KnetArray{T}}
const WeightParams{T} = Dict{String, <:ArrayOrKnetArray{T}}
const WeightParamsR{T} = Union{<:WeightParams{T}, <:AutoGrad.Rec{<:WeightParams{T}}}

function predict_q(w::WeightParamsR{T}, x::VecOrMat{T}; nh::Int=1) where {T}
    inp = x
    for i=1:nh
        inp = relu.(w["w_$i"] * inp .+ w["b_$i"])
    end
    q = w["w_out"] * inp .+ w["b_out"]
    return q
end

function init_weights(input::Int, hiddens::Vector{Int}, nout::Int, atype::Type{<:KnetArray})
    w0 = Dict{String, Any}()
    inp = input
    for i=1:length(hiddens)
        w0["w_$i"] = xavier(hiddens[i], inp)
        w0["b_$i"] = zeros(hiddens[i])
        inp = hiddens[i]
    end

    w0["w_out"] = xavier(nout, hiddens[end])
    w0["b_out"] = zeros(nout, 1)

    return Dict{String, atype}(k => convert(atype, w0[k]) for k in keys(w0))
end

function init_weights(input::Int, hiddens::Vector{Int}, nout::Int, ::Type{<:Array{T}}) where T
    w = Dict{String, Array{T}}()
    inp = input
    for i=1:length(hiddens)
        w["w_$i"] = xavier(T, hiddens[i], inp)
        w["b_$i"] = zeros(T, hiddens[i])
        inp = hiddens[i]
    end

    w["w_out"] = xavier(T, nout, hiddens[end])
    w["b_out"] = zeros(T, nout, 1)

    return w
end

function save_model(w::WeightParams{T}, fname::AbstractString) where {T}
    tmp = Dict{String, Array{T}}()
    for k in keys(w)
        tmp[k] = convert(Array{T}, w[k])
    end
    save(fname, "model", tmp)
end

function load_model(fname::AbstractString, ::Type{A}) where {A<:ArrayOrKnetArray{T} where T}
    w = load(fname, "model")
    return Dict{String, A}(k => convert(A, w[k]) for k in keys(w))
end

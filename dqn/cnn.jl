# cnn.jl
# original: https://github.com/denizyuret/Knet.jl/blob/master/examples/reinforcement-learning/dqn/mlp.jl
# refer-to: https://github.com/denizyuret/Knet.jl/blob/master/examples/cifar10-cnn/cnn_batchnorm.jl

using AutoGrad

const ArrayOrKnetArray{T} = Union{<:Array{T}, <:KnetArray{T}}
const WeightParams{T} = Dict{String, <:ArrayOrKnetArray{T}}
const WeightParamsR{T} = Union{<:WeightParams{T}, <:AutoGrad.Rec{<:WeightParams{T}}}

#=
Initialization is from
He et al., 2015, 
Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
https://arxiv.org/abs/1502.01852
=#
kaiming(et, h, w, i, o) = et(sqrt(2 / (w * h * o))) .* randn(et, h, w, i, o)

function predict_q(w::WeightParamsR{T}, x::Array{T}; nc::Int=3, nh::Int=1) where {T}
    inp = x
    for i=1:nc
        inp = relu.(conv4(w["w_$i"], inp; padding=1) .+ w["b_$i"])
        inp = pool(inp; window=3, stride=2, padding=1)
    end
    inp = reshape(inp, size(w["w_$(nc+1)"], 2), :)
    for i=nc+1:nc+nh
        inp = relu.(w["w_$i"] * inp .+ w["b_$i"])
    end
    q = w["w_out"] * inp .+ w["b_out"]
    return q
end

# function init_weights(input::Int, hiddens::Vector{Int}, nout::Int, atype::Type{<:KnetArray})
#     w0 = Dict{String, Any}()
#     inp = input
#     for i=1:length(hiddens)
#         w0["w_$i"] = xavier(hiddens[i], inp)
#         w0["b_$i"] = zeros(hiddens[i])
#         inp = hiddens[i]
#     end

#     w0["w_out"] = xavier(nout, hiddens[end])
#     w0["b_out"] = zeros(nout, 1)

#     return Dict{String, atype}(k => convert(atype, w0[k]) for k in keys(w0))
# end

function init_weights(input_size::NTuple{3, Int}, conv_channels::Vector{Int}, hiddens::Vector{Int}, nout::Int, ::Type{<:Array{T}}) where T
    w = Dict{String, Array{T}}()
    ht, wd, ch = input_size
    nc = length(conv_channels)
    for i=1:nc
        next_ch = conv_channels[i]
        w["w_$i"] = kaiming(T, 3, 3, ch, next_ch)
        w["b_$i"] = zeros(T, 1, 1, next_ch)
        ht = cld(ht, 2)
        wd = cld(wd, 2)
        ch = next_ch
    end
    inp = ht * wd * ch
    for i=nc+1:nc+length(hiddens)
        w["w_$i"] = xavier(T, hiddens[i-nc], inp)
        w["b_$i"] = zeros(T, hiddens[i-nc])
        inp = hiddens[i-nc]
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

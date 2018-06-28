# dqn_cnn.jl
# original: https://github.com/denizyuret/Knet.jl/blob/master/examples/reinforcement-learning/dqn/dqn_with_target.jl

module DQN

using Gym, Knet, JLD

include("replay_buffer.jl")
include("cnn.jl")
include("piecewise_schedule.jl")

function loss(w::WeightParamsR{T}, states::Array{T}, actions, targets; nc=3, nh=1) where {T}
    qvals = predict_q(w, states; nc=nc, nh=nh)
    nrows = size(qvals, 1)
    index = actions .+ nrows .* (0:length(actions)-1)
    qpred = reshape(qvals[index], size(targets))
    mse = sum(abs2, targets .- qpred) / size(states, 2)
    return mse
end

lossgradient = gradloss(loss)

function train!(w::WeightParams{T}, prms, states, actions, targets; nc=3, nh=1) where {T}
    g, mse = lossgradient(w, states, actions, targets; nc=nc, nh=nh)
    update!(w, g, prms)
    return mse
end

function n4max(img::Array{T,2}, newsize::NTuple{2,Int}) where {T}
    ih, iw = size(img)
    oh, ow = newsize
    newimg = zeros(T, oh, ow)
    for oy = 1:oh
        for ox = 1:ow
            ix = fld((ox - 1) * (iw - 2), (ow - 1)) + 1
            iy = fld((oy - 1) * (ih - 2), (oh - 1)) + 1
            newimg[oy, ox] = maximum(img[iy:iy+1, ix:ix+1])
        end
    end
    newimg
end

function adjust_obs(ob_t::Array{UInt8, 3})
    H, W, _ = size(ob_t)
    float_ob_t = ob_t ./ 255f0
    gray_ob_t = Float32[0.299f0*float_ob_t[y, x, 1] + 0.587f0*float_ob_t[y, x, 2] + 0.114f0*float_ob_t[y, x, 3]
                        for y=1:H, x=1:W]
    resized_ob_t = n4max(gray_ob_t, (64, 64))
    reshape(resized_ob_t, (64, 64, 1))
end

function adjust_obs(t::Tuple{Array{UInt8, 3}, Any, Any, Any})
    ob_t, _reward, _done, _info = t
    (adjust_obs(ob_t), _reward, _done, _info)
end

# adjust_obs(ob_t) = ob_t # FALLBACK

function dqn_learn(w::WeightParams{T}, opts, env, buffer, exploration, o) where {T}
    total = 0.0
    readytosave = save_interval = get(o, "save_interval", 10000)
    episode_rewards = Float32[]
    frames = Float32[]
    ob_t = adjust_obs(reset!(env))

    n_convs = get(o, "n_convs", length(o["channels"]))::Int
    n_hiddens = get(o, "n_hiddens", length(o["hiddens"]))::Int
    target_w = o["play"] || o["tupdate"] <= 1 ? w : deepcopy(w)

    for fnum = 1:o["frames"]
        o["render"] && render(env)
        #process the raw ob
        ob_t_reshaped = reshape(ob_t, size(ob_t)..., 1)
        if !o["play"] && rand() < value(exploration, fnum)
            a = sample(env.action_space)
        else
            obses_t = encode_recent(buffer, ob_t_reshaped; stack=o["stack"])
            inp = convert(o["atype"], obses_t)
            qvals = predict_q(w, inp; nc=n_convs, nh=n_hiddens)
            a = indmax(Array(qvals)) - 1
        end
        
        ob_t, reward, done, _ = adjust_obs(step!(env, a))
        total += reward 

        if !o["play"]
            #process the raw ob
            ob_tp1_reshaped = reshape(ob_t, size(ob_t)..., 1)
            push!(buffer, ob_t_reshaped, a + 1, reward, ob_tp1_reshaped, done)
            
            if can_sample(buffer, o["bs"])
                obses_t, actions, rewards, obses_tp1, dones = sample_batch(buffer, o["bs"]; stack=o["stack"])
                obses_tp1 = convert(o["atype"], obses_tp1)
                #predict next q values with the target network
                nextq = Array(predict_q(target_w, obses_tp1; nc=n_convs, nh=n_hiddens))
                maxs = maximum(nextq, 1)
                nextmax = sum(nextq .* (nextq .== maxs), 1)
                targets = reshape(rewards, 1, :) .+ (o["gamma"] .* nextmax .* dones)
                mse = train!(w, opts, convert(o["atype"], obses_t), actions, convert(o["atype"], targets); nc=n_convs, nh=n_hiddens)
            end

            if !o["no_save"] && fnum > readytosave
                model_filepath = "$(get(o, "model_prefix", ""))_$(fnum).jld"
                save_model(w, model_filepath, n_convs, n_hiddens)
                readytosave += save_interval
            end

            if o["tupdate"] > 1 && fnum % o["tupdate"] == 0
                target_w = deepcopy(w)
            end
        end

        if done
            ob_t = adjust_obs(reset!(env))
            # o["printinfo"] && println("Frame: $fnum , Total reward: $total, Exploration Rate: $(value(exploration, fnum))")
            if o["printinfo"]
                if o["play"]
                    println("Frame: $fnum , Total reward: $total")
                else
                    println("Frame: $fnum , Total reward: $total, Exploration Rate: $(value(exploration, fnum))")
                end
            end
            push!(episode_rewards, total)
            push!(frames, fnum)
            total = 0.0
        end
    end
    return episode_rewards, frames
end

end # module
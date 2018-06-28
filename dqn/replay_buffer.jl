# replay_buffer.jl
# original: https://github.com/denizyuret/Knet.jl/blob/master/examples/reinforcement-learning/dqn/replay_buffer.jl

import Base.length
import Base.push!

"""
Replay buffer implementation based on https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
"""
mutable struct ReplayBuffer
    size::Int
    storage::Vector{Any}
    next_idx::Int
end

"""
Constructor
"""
ReplayBuffer(size::Int) = ReplayBuffer(size, Any[], 1)

"""
Length of the buffer
"""
length(buf::ReplayBuffer) = length(buf.storage)

"""
Tells whether there is enough data in the buffer or not
"""
can_sample(buf::ReplayBuffer, batch_size::Int) = batch_size <= length(buf)

"""
Add new transition to the buffer
"""
function push!(buf::ReplayBuffer, obs_t, action, reward, obs_tp1, done)
    data = (obs_t, action, reward, obs_tp1, done)

    if buf.next_idx > length(buf)
        push!(buf.storage, data)
    else
        buf.storage[buf.next_idx] = data
    end

    buf.next_idx = mod1(buf.next_idx + 1, buf.size)
end

function encode_sample(buf::ReplayBuffer, idxes; stack=1)
    bs = length(idxes)
    inpsize = size(buf.storage[1][1])[1:(end-1)] # get the dimensions of the input
    stacksize = nothing
    if length(inpsize) < 3
        stacksize = (inpsize[1]*stack, inpsize[2:end]...)
    else
        stacksize = (inpsize[1], inpsize[2], inpsize[3]*stack, inpsize[4:end]...)
    end

    obses_t = zeros(Float32, stacksize..., bs)
    actions = zeros(Int, bs)
    rewards = zeros(Float32, bs)
    obses_tp1 = zeros(Float32, stacksize..., bs)
    dones = zeros(Float32, 1, bs)

    indx = 0
    for ind in idxes
        indx += 1
        for i=-(stack-1):0
            curr = ind + i
            curr = curr < 1 ? 1 : curr
            data = buf.storage[curr] #use available data instead of filling the history with 0

            obs_t, action, reward, obs_tp1, done = data
            if length(inpsize) < 3
                t_t = size(obs_t, 1)
                obses_t[t_t*(i-1+stack)+1:t_t*(i-1+stack)+t_t, (Colon() for _=2:ndims(obs_t)-1)..., indx] = obs_t #stack frames
                t_tp1 = size(obs_tp1, 1)
                obses_tp1[t_tp1*(i-1+stack)+1:t_tp1*(i-1+stack)+t_tp1, (Colon() for _=2:ndims(obs_tp1)-1)..., indx] = obs_tp1 #stack frames
            else
                t_t = size(obs_t, 3)
                obses_t[:, :, t_t*(i-1+stack)+1:t_t*(i-1+stack)+t_t, (Colon() for _=4:ndims(obs_t)-1)..., indx] = obs_t #stack frames
                t_tp1 = size(obs_tp1, 3)
                obses_tp1[:, :, t_tp1*(i-1+stack)+1:t_tp1*(i-1+stack)+t_tp1, (Colon() for _=4:ndims(obs_tp1)-1)..., indx] = obs_tp1 #stack frames
            end
            if i == 0
                actions[indx] = action
                rewards[indx] = reward
                dones[1, indx] = done ? 0.0 : 1.0
            end
        end
    end
    obses_t, actions, rewards, obses_tp1, dones
end

function encode_recent(buf::ReplayBuffer, obs_t; stack::Int=1)
    if length(buf) != 0
        recent_idx = max(1, (buf.next_idx - 1) % (buf.size+1))
        obses_t, _, _, _,_ = encode_sample(buf, [recent_idx]; stack=stack)
        inpsize = size(obs_t)[1:(end-1)] # get the dimensions of the input
        i = 0
        if length(inpsize) < 3
            t = size(obs_t, 1)
            obses_t[t*(i-1+stack)+1:t*(i-1+stack)+t, (Colon() for _=2:ndims(obs_t)-1)..., 1] = obs_t #stack frames
        else
            t = size(obs_t, 3)
            obses_t[:, :, t*(i-1+stack)+1:t*(i-1+stack)+t, (Colon() for _=4:ndims(obs_t)-1)..., 1] = obs_t #stack frames
        end
    else
        bs = 1
        inpsize = size(obs_t)[1:end-1] # get the dimensions of the input
        stacksize = nothing
        if length(inpsize) < 3
            stacksize = (inpsize[1]*stack, inpsize[2:end]...)
        else
            stacksize = (inpsize[1], inpsize[2], inpsize[3]*stack, inpsize[4:end]...)
        end

        obses_t = zeros(Float32, stacksize..., bs)
        for i = -(stack-1):0
            if length(inpsize) < 3
                t = size(obs_t, 1)
                obses_t[t*(i-1+stack)+1:t*(i-1+stack)+t, (Colon() for _=2:ndims(obs_t)-1)..., 1] = obs_t #stack frames
            else
                t = size(obs_t, 3)
                obses_t[:, :, t*(i-1+stack)+1:t*(i-1+stack)+t, (Colon() for _=4:ndims(obs_t)-1)..., 1] = obs_t #stack frames
            end
        end
    end
    return obses_t
end

function sample_batch(buf::ReplayBuffer, batchsize::Int; stack::Int=1)
    idxes = randperm(length(buf))[1:batchsize]
    return encode_sample(buf, idxes; stack=stack)
end
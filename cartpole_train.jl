# cartpole_train_sample.jl
using Gym
using Knet
using ArgParse

include(joinpath(@__DIR__, "dqn", "dqn.jl"))

main() = main(ARGS)
main(args::String) = main(split(args))
function main(args::Vector{<:AbstractString})
    s = ArgParseSettings()
    # s.description = "(c) Ozan Arkan Can, 2018. An implementation of the deep q network."
    @add_arg_table s begin
        ("--frames"; arg_type=Int; default=30001; help="number of frames")
        ("--lr"; arg_type=Float64; default=0.001; help="learning rate")
        ("--gamma"; arg_type=Float64; default=0.99; help="discount factor")
        ("--hiddens"; arg_type=Int; nargs='+'; default=[32]; help="number of units in the hiddens for the mlp")
        ("--env_id"; default="CartPole-v0")
        ("--render"; action=:store_true)
        ("--memory"; arg_type=Int; default=10000; help="memory size")
        ("--bs"; arg_type=Int; default=32; help="batch size")
        ("--stack"; arg_type=Int; default=4; help="length of the frame history")
        # ("--save"; help="model name")
        ("--load"; help="load model name")
        ("--save_interval"; arg_type=Int; default=10000; help="save interval (per frames)")
        ("--no_save"; action=:store_true; help="to prevent to save the model")
        ("--model_prefix"; arg_type=String; help="model name prefix (default: YYYYmmddHHMMSS)")
        ("--play"; action=:store_true; help="only play")
        ("--printinfo"; action=:store_true; help="print the training messages")
        ("--tupdate"; arg_type=Int; default=500; help="update frequency for the target network")
    end
    if "--help" ∈ args || "-h" ∈ args
        ArgParse.show_help(s; exit_when_done=false)
        return
    end
    o = parse_args(args, s)
    o["atype"] = Array{Float32}
    # o["play"] = false

    if o["model_prefix"] === nothing
        o["model_prefix"] = Dates.format(Dates.now(), Dates.dateformat"YYYYmmddHHMMSS")
    end

    # train
    env = GymEnv(o["env_id"])
    # seed!(env, 12345)
    # cartpoleenv = env.gymenv
    # cartpoleenv[:_max_episode_steps] = typemax(Int)
    # cartpoleenv[:_max_episode_seconds] = typemax(Int)

    if o["load"] === nothing
        INPUT = env.observation_space.shape[1] * o["stack"]
        OUTPUT = env.action_space.n
        w = DQN.init_weights(INPUT, o["hiddens"], OUTPUT, o["atype"])
        o["n_hiddens"] = length(o["hiddens"])
    else
        w = DQN.load_model(o["load"], o["atype"])
        o["n_hiddens"] = length(w) ÷ 2 - 1
    end

    opts = Dict(k => Rmsprop(lr=o["lr"]) for k in keys(w))
    buffer = DQN.ReplayBuffer(o["memory"])
    exploration = DQN.PiecewiseSchedule([(0, 1.0),
        (round(Int, o["frames"]/5), 0.1)])
    # exploration = DQN.PiecewiseSchedule([(0, 1.0), (min(o["frames"] ÷ 3, 10000), 0.1)])

    rewards, frames = DQN.dqn_learn(w, opts, env, buffer, exploration, o)

    close!(env)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

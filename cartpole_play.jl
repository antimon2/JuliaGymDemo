# cartpole_play.jl
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
        ("model"; arg_type=String; help="model name"; required=true)
        ("--frames"; arg_type=Int; default=2000; help="number of frames")
        # ("--lr"; arg_type=Float64; default=0.001; help="learning rate")
        # ("--gamma"; arg_type=Float64; default=0.99; help="discount factor")
        # ("--hiddens"; arg_type=Int; nargs='+'; default=[32]; help="number of units in the hiddens for the mlp")
        ("--env_id"; default="CartPole-v0")
        ("--render"; action=:store_true)
        # ("--memory"; arg_type=Int; default=10000; help="memory size")
        # ("--bs"; arg_type=Int; default=32; help="batch size")
        ("--stack"; arg_type=Int; default=4; help="length of the frame history")
        # ("--save"; default=""; help="model name")
        # ("--load"; default=""; help="model name")
        # ("--play"; action=:store_true; help="only play")
        ("--printinfo"; action=:store_true; help="print the playing messages")
        # ("--tupdate"; arg_type=Int; default=500; help="update frequency for the target network")
    end
    if "--help" ∈ args || "-h" ∈ args
        ArgParse.show_help(s; exit_when_done=false)
        return
    end
    o = parse_args(args, s)
    o["atype"] = Array{Float32}
    # o["load"] = o["model"]
    o["play"] = true

    env = GymEnv(o["env_id"])
    cartpoleenv = env.gymenv
    cartpoleenv[:_max_episode_steps] = typemax(Int)
    cartpoleenv[:_max_episode_seconds] = typemax(Int)

    w = DQN.load_model(o["model"], o["atype"])
    o["n_hiddens"] = length(w) ÷ 2 - 1
    o["hiddens"] = [0]  # DUMMY
    opts = Dict{String, Array{Float32}}()   # DUMMY
    buffer = DQN.ReplayBuffer(1)    # DUMMY
    exploration = DQN.PiecewiseSchedule([(0, 1.0)]) # DUMMY

    _rewards, _frames = DQN.dqn_learn(w, opts, env, buffer, exploration, o)

    close!(env)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

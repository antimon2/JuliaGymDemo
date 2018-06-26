# cartpole_sample.jl
using Gym

env = GymEnv("CartPole-v0")

action_space = env.action_space

# @show action_space.n

obs_space = env.observation_space

# @show obs_space.shape

reward = 0.0
episode_count = 10

for i=1:episode_count
    total = 0
    ob = reset!(env)
    render(env)#comment out this line if you do not want to visualize the environment
    while true
        action = sample(env.action_space)
        ob, reward, done, information = step!(env, action)
        total += reward
        render(env)#comment out this line if you do not want to visualize the environment
        done && break
    end
    println("episode $i total Rewards: $total")
end

close!(env)
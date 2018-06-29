# JuliaGymDemo
Train and Play Gym with Julia v0.6 + Gym.jl(revised) + Knet.jl

## Prerequisites

What things you need to install the software and how to install them

### Target Julia and Packages 

+ Julia v0.6.x
    + Gym.jl
    + ArgParse.jl
    + JLD.jl
    + Knet.jl

### Simple Installation

Run the codes on Julia REPL:

```julia
julia> Pkg.clone("https://github.com/antimon2/Gym.jl.git")
julia> Pkg.checkout("Gym", "mln_ngy")
julia> Pkg.build("Gym")
julia> Pkg.add("ArgParse")
julia> Pkg.add("JLD")
julia> Pkg.add("Knet")
```

### Custom Installation with installed `Gym`

If you have already installed `Gym` on your own Python (for instance, on the path s.t. `/path/to/user_home/.pyenv/versions/3.6.x/bin/python`):

```julia
julia> ENV["PYTHON"] = "/path/to/user_home/.pyenv/versions/3.6.x/bin/python"
julia> Pkg.clone("https://github.com/antimon2/PyCall.jl.git")
julia> Pkg.checkout("PyCall", "mln_ngy")
julia> Pkg.clone("https://github.com/antimon2/Gym.jl.git")
julia> Pkg.checkout("Gym", "mln_ngy")
julia> Pkg.build("Gym")
julia> Pkg.add("ArgParse")
julia> Pkg.add("JLD")
julia> Pkg.add("Knet")
```

## Getting Started

Clone this repository, run `julia cartpole_sample.jl`:

```bash
$ git clone https://github.com/antimon2/JuliaGymDemo
$ cd JuliaGymDemo
$ julia julia cartpole_sample.jl
WARN: gym.spaces.Box autodetected dtype as <class 'numpy.float32'>. Please provide explicit dtype.
episode 1 total Rewards: 21.0
episode 2 total Rewards: 13.0
episode 3 total Rewards: 48.0
episode 4 total Rewards: 16.0
episode 5 total Rewards: 15.0
episode 6 total Rewards: 14.0
episode 7 total Rewards: 46.0
episode 8 total Rewards: 12.0
episode 9 total Rewards: 14.0
episode 10 total Rewards: 33.0
$ 
```

Then you can see outputs like above and window like below:

![表示例](https://i.imgur.com/tIFDr8R.png)

## for Trainings and Tests

+ `cartpole_train.jl`: Train the Gym Env `CartPole-v0`
    + `julia cartpole_train.jl -h` for help.
+ `cartpole_play.jl`: Play the Gym Env `CartPole-v0` with trained model saved by `cartpole_train.jl`
    + `julia cartpole_play.jl -h` for help.

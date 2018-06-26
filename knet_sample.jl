# knet_sample.jl
using Knet

n = 100
x0 = linspace(-2,2,n)
a0 = 3.0
a1= 2.0
b0 = 1.0
y0 = zeros(Float32,(1,n))
f(x0) = a0.*x0 + a1.*x0.^2 + b0 + 3*cos.(20*x0)
y0[:] = f(x0)

function make_φ(x0,n,k)
    φ = zeros(Float32,k,n)
    for i in 1:k
        φ[i,:] = x0.^(i-1)
    end
    return φ
end
k = 4
φ = make_φ(x0,n,k)

w = Any[ones(Float32,(1,k)),ones(Float32,1)]

function predict(w,x)
    y = w[1]*x .+w[2]
    return y
end

loss(w,x,y) = mean(abs2,y-predict(w,x))
lossgradient = grad(loss)

function train(model, data, optim)
    for (x,y) in data
        grads = lossgradient(model,x,y)
        update!(model, grads, optim)
    end
end

dtrn = minibatch(φ,y0,10,shuffle=true)

o = optimizers(w, Adam)
for i in 1:2000
    train(w,dtrn,o)
    if i%100 == 0 
        println(loss(w,φ,y0))
    end
end

ye = predict(w,φ)

@show ye
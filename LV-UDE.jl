using JLD, Lux, DiffEqFlux, DifferentialEquations, Optimization, OptimizationOptimJL, Random, Plots
using ComponentArrays
using OptimizationOptimisers

# Generate data which simulates number of rabbits and fox over the time

# Lotka-Volterra Equation

function LV(du,u,p,t)
    (x,y) = u
    (α,β,δ,γ) = p
    du[1] = p[1]*u[1] - p[2]*u[1]*u[2]
    du[2] = -p[3]*u[2] + p[4]*u[1]*u[2]
end


# Generating random initial value for 🐰 & 🦊 AND the α,β,δ,γ parameters AND time
rng = Random.default_rng()

u0 = [1.0, 1.0]


# Random values of α,β,δ,γ 
α = 1.5
β = 1.0
δ = 1.0
γ = 3.0

p = [α,β,δ,γ]
tspan = [0.0,5.0]
t = range(0.0, 5.0, step=0.25)

# ODE
prob = ODEProblem(LV, u0, tspan, p)
solution = solve(prob, Tsit5(), saveat=t)

# Plot
plot(solution, xlabel="Time", ylabel="Population", label=["Prey" "Predator"],
     title="Lotka-Volterra Predator-Prey Model")



rabbits_data = Array(solution)[1,:]
foxes_data = Array(solution)[2,:]

# Creating UDE

p0_vec = []

# αx
NN1 = Lux.Chain(Lux.Dense(1,10,relu),Lux.Dense(10,1))
p1, st1 = Lux.setup(rng, NN1)

# βxy or γxy
NN2 = Lux.Chain(Lux.Dense(2,10,relu),Lux.Dense(10,2))
p2, st2 = Lux.setup(rng, NN2)

# δy
NN3 = Lux.Chain(Lux.Dense(1,10,relu),Lux.Dense(10,1))
p3, st3 = Lux.setup(rng, NN3)


p0_vec = (layer_1 = p1, layer_2 = p2, layer_3 = p3)
p0_vec = ComponentArray(p0_vec)


function UDE(du, u, p, t)
    (x,y) = u

    NNx = abs(NN1([x], p.layer_1, st1)[1][1])
    NNxy = abs(NN2([x,y], p.layer_2, st2)[1][1])
    NNy = abs(NN3([y], p.layer_3, st3)[1][1])
    

    du[1] = NNx - NNxy
    du[2] = -NNy + NNxy 
end

prob_pred = ODEProblem{true}(UDE,u0,tspan)


function predict_adjoint(θ)
  x = Array(solve(prob_pred,Tsit5(),p=θ,saveat=t,
                  sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))))
end


function loss_adjoint(θ)
    x = predict_adjoint(θ)
    loss =  sum( abs2, (rabbits_data .- x[1,:])[2:end])
    loss += sum( abs2, (foxes_data .- x[2,:])[2:end])
    return loss
end

iter = 0
function callback2(θ,l)
  global iter
  iter += 1
  if iter%250 == 0
    println(l)
  end
  return false
end

# Initialize global variables
iter = 0
loss_history = Float64[]

# Define the callback function
function callback2(θ, l)
    global iter, loss_history
    iter += 1
    push!(loss_history, l)  # Store the loss value in the history array

    # Print iteration number and loss every 25 iterations
    if iter % 25 == 0
        println("Iteration $iter, Loss: $l")
    end
    return false
end


adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x,p) -> loss_adjoint(x), adtype)
optprob = Optimization.OptimizationProblem(optf, p0_vec)
res1 = Optimization.solve(optprob, OptimizationOptimisers.ADAM(0.001), callback = callback2, maxiters = 3000)

# After training is complete, plot the loss curve
function plot_loss_curve()
  plot(1:iter, loss_history, xlabel="Iteration", ylabel="Loss",
       title="Loss Curve", label="Training Loss", lw=2, color=:blue)
end

plot_loss_curve()



# Visualizing the predictions
data_pred = predict_adjoint(res1.u)
plot( legend=:topright)
  
# Bar plots for observed data
bar!(t, rabbits_data, label="Rabbits Data", color=:red, alpha=0.5, legend=:topright)
bar!(t, foxes_data, label="Foxes Data", color=:blue, alpha=0.5)

# Line plots for predicted data
plot!(t, data_pred[1, :], label="Rabbits Prediction", color=:red, lw=2)
plot!(t, data_pred[2, :], label="Foxes Prediction", color=:blue, lw=2)

# Adding title and axis labels
title!("Predator-Prey Model: Data and Prediction Comparison")
xlabel!("Time (t)")
ylabel!("Population")

# Display the plot
plot!()

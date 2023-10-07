function LxF(Q::Array, n, i, f::Function)
    return 1.0/2.0 * (Q[n, i-1] + Q[n, i+1]) - t/(2.0*h) * (f(Q[n, i+1]) - f(Q[n, i-1]))
end

function LeftBorderCondition(n)
    return [1.0, 0, 1.0 / (gamma - 1)]
end

function RightBorderCondition(n)
    return [0.125, 0, 0.1 / (gamma - 1)]
end

function InitialCondition(i)
    if (i-1) * h < 0.5
        return [1.0, 0, 1.0 / (gamma - 1)]
    else
        return [0.125, 0, 0.1 / (gamma - 1)]
    end
end

function f(q::Array)
    # rho =  q[2]^2/q[1]
    rho = q[1]
    u = q[2]/rho
    etot = q[3]
    eint = etot - 1.0/2.0 * rho * u^2
    p = eint * (gamma - 1)
    return [ q[2], 
             rho * u^2  + p,
             (etot + p) * u ]
end

X = 1.0 # Dimension coordinate boundary [meters]
T = 0.15 # Time boundary [seconds]

xSteps = 500 # Number of steps on dimension coordinate
tSteps = 2000 # Number of steps on time

h = X / xSteps # Step on dimension coordinate
t = T / tSteps # Step on time
gamma = 1.4 # Polytropic index

Q = fill(Array{Float64}(undef, 3) , tSteps, xSteps)

for i in axes(Q, 2)
    Q[1, i] = InitialCondition(i)
end

for n in 2:size(Q)[1]
    Q[n, 1] = LeftBorderCondition(n)
    Q[n, size(Q, 2)] = RightBorderCondition(n)
    for i in 2:size(Q, 2)-1
        Q[n, i] = LxF(Q, n-1, i, f)
    end
end

# for i in axes(Q, 2)
#     print(Q[end, i][1], " ")
# end

using Plots

x = [h * i for i in 0:xSteps-1]
y = [Q[end, i][1] for i in axes(Q, 2)]
plot(x, y)
savefig("pho.png")

x = [h * i for i in 0:xSteps-1]
y = [Q[end, i][2] for i in axes(Q, 2)]
plot(x, y)
savefig("phou.png")

x = [h * i for i in 0:xSteps-1]
y = [Q[end, i][3] for i in axes(Q, 2)]
plot(x, y)
savefig("E.png")



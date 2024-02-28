#
# hyperelasticitymph_postproc.jl
#

# TODO: To implement it!


x = [dx * i for i in 0:nx-1]
alpha, true_den, vel, A, ent, info = postproc_arrays(eos, Q0)

# ---
# Fraction
plot(x, alpha[1, :], label = L"Fraction")

xlabel!(L"$x$")
ylabel!(L"$\alpha$")
title!("Fraction of [1]")

fname = "fraction.png"
savefig(joinpath(dname, fname))

# ---
# Density
plot(x, alpha[1, :] .* true_den[1, :], label = L"\rho")

xlabel!(L"$x$")
ylabel!(L"$\rho$")
title!("Density")

fname = "density.png"
savefig(joinpath(dname, fname))

# ---
# Velocity
for i in 1:3
    lbl = @sprintf("\$v_%i\$", i)
    plot(x, vel[i, 1, :], label =  lbl)

    xlabel!(L"$x$")
    ylabel!(lbl)
    title!("Velocity")    
    
    global fname = @sprintf("velocity_%i.png", i)
    savefig(joinpath(dname, fname))
end

#---
# Stress
for i in 1:6
    lbl = @sprintf("\$A_{%i%i}\$", i % 3, div(i, 3))

    plot(x, A[i, 1, :], label = lbl)
    
    xlabel!(L"$x$")
    ylabel!(lbl)
    title!("Distortion")

    global fname = @sprintf("distortion_%i%i", i % 3, div(i, 3))
    savefig(joinpath(dname, fname))
end

# ---
# Deformation gradient

# ---
plot(x, ent[1, :], label = L"$\eta$")

xlabel!(L"$x$")
ylabel!(L"$\eta$")
title!("Entropy")

fname = @sprintf("entropy.png")
savefig( joinpath(dname, fname) )

# EOF

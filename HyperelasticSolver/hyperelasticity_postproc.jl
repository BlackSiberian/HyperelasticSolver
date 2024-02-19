#
# hyperelasticity_postproc.jl
#

x = [dx * i for i in 0:nx-1]
den, ent, vel, strs, eint, info = postproc_arrays(Q0)

# ---
# Density
plot(x, den, label = L"\rho")

xlabel!(L"$x$")
ylabel!(L"$\rho$")
title!("Density")

fname = "density.png"
savefig(joinpath(dname, fname))

# ---
# Velocity
for i in 1:3
    lbl = @sprintf("\$v_%i\$", i)
    plot(x, vel[i, :], label =  lbl)

    xlabel!(L"$x$")
    ylabel!(lbl)
    title!("Velocity")    
    
    global fname = @sprintf("velocity_%i.png", i)
    savefig(joinpath(dname, fname))
end

#---
# Stress
for i in 1:3
    for j in i:3

        lbl = @sprintf("\$\\sigma_{%i%i}\$", i,j)
        
        plot(x, strs[i, j, :], label = lbl)
        
        xlabel!(L"$x$")
        ylabel!(lbl)
        title!("Stress")
        
        global fname = @sprintf("stress_%i%i.png", i,j)
        savefig( joinpath(dname, fname) )
    end
end

# ---
# Deformation gradient

# ---
plot(x, ent, label = L"$\eta$")


xlabel!(L"$x$")
ylabel!(L"$\eta$")
title!("Entropy")

fname = @sprintf("entropy.png")
savefig( joinpath(dname, fname) )


# EOF
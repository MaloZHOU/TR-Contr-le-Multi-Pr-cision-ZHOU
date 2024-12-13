include("goddard_hermit-Simpson.jl")
include("goddard.jl")

function Compa_nh_tol(nh,tol)
    p = Plots.plot(title = "Comparaison des courbes, nh = $nh, tol = $tol",xlabel="Temps", ylabel="Value")
    model1 = rocket_model(nh)
    model2 = rocket_model_hersim(nh)
    JuMP.set_optimizer(model1, Ipopt.Optimizer)
    JuMP.set_optimizer(model2, Ipopt.Optimizer)
    JuMP.set_attribute(model1, "tol", tol)
    JuMP.set_attribute(model2, "tol", tol)
    JuMP.optimize!(model1)
    JuMP.optimize!(model2)
    Thrust1 = collect(JuMP.value.(model1[:T]))[:, 1]
    Thrust2 = collect(JuMP.value.(model2[:T]))[:, 1]
    Plots.plot!(LinRange(0,1,length(Thrust1)),Thrust1,label="T values from Trapezoidal")
    Plots.plot!(LinRange(0,1,length(Thrust2)),Thrust2,label="T values from Hermite-Simpson")
    #Plots.display(p)
    return p
end

nhs = [100,1000,5000,10000]
tols = [1e-8,1e-10,1e-12]
for nh in nhs
    for tol in tols
        p = Compa_nh_tol(nh,tol)
        Plots.savefig("Photo/nh=$nh,tol=$tol.png")
    end
end
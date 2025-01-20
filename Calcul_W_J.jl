include("goddard_hermite-Simpson.jl")
include("goddard.jl")
include("goddard_Euler.jl")
using NLPModels
using NLPModelsIpopt
using NLPModelsJuMP
using SparseArrays
using SuiteSparse
using Plots
using LinearAlgebra

function normalize_matrix(mat)
    min_val = minimum(mat)
    max_val = maximum(mat)
    (mat .- min_val) ./ (max_val - min_val)
end

function present_Heatmap(Mat,name="")
    norm_matrix = normalize_matrix(Mat)
    max_Mat = round(maximum(Mat),sigdigits = 4)
    min_Mat = round(minimum(Mat),sigdigits = 4)
    Lims = "[$min_Mat,$max_Mat]"
    p = heatmap(norm_matrix, color=:grays, 
            xlabel="Columns", ylabel="Rows", title="Matrix Heatmap of H $name")
    Plots.savefig(p,"$name,$Lims.png")
    return p
end

function Calcul_H_model(goddard_model,name_model,nh=100)
    model = goddard_model(nh)
    nlp = MathOptNLPModel(model)
    res = ipopt(nlp)
    x = res.solution
    y = res.multipliers
    n = NLPModels.get_nvar(nlp)
    m = NLPModels.get_ncon(nlp)
    
    nnzh = NLPModels.get_nnzh(nlp)
    Wi, Wj = NLPModels.hess_structure(nlp)
    Wz = zeros(nnzh)
    NLPModels.hess_coord!(nlp, x, y, Wz)
    W = sparse(Wi, Wj, Wz, n, n)

    nnzj = NLPModels.get_nnzj(nlp)
    Ji, Jj = NLPModels.jac_structure(nlp)
    Jz = zeros(nnzj)
    NLPModels.jac_coord!(nlp, x, Jz)
    J = sparse(Ji, Jj, Jz, m, n)

    j1 = Matrix(J)
    Z = nullspace(j1)
    H = Z'*Symmetric(W, :L)*Z    
    present_Heatmap(H,name_model)

    return H
end

function Calcul_eigvals_models(models,names,nh = 100)
    Xs = []
    Ys = []
    for i in range(1,length(models))
        H=Calcul_H_model(models[i],names[i],nh)
        Eigs = eigvals(H)
        append!(Xs,Eigs)
        append!(Ys,[i for n in range(1,length(Eigs))])
    end
    Plots.scatter(Ys,Xs,xticks = (1:length(names),names),label = false)
    Plots.savefig("Variance in Eig,scatter, nh= $nh.png")
end

function Calcul_eigvals_euler(nh = 100,tols = [1e-6])
    ###Generate the additional lines in J_act
    ###Paramaters
    model = rocket_model(nh)
    nlp = MathOptNLPModel(model)
    res = ipopt(nlp)
    
    ind_h = 1:101
    ind_v = 102:202
    ind_m = 203:303
    ind_T = 304:404
    Lim_l_h = 1.0
    Lim_l_v = 0.0
    Lim_l_m = 0.6
    Lim_l_T = 0.0
    Lim_u_m = 1.0
    Lim_u_T = 3.5*1.0*1.0
    
    I_l = zeros(4,405)
    I_u = zeros(2,405)
    
    sol_h = res.solution[ind_h]
    sol_v = res.solution[ind_v]
    sol_m = res.solution[ind_m]
    sol_T = res.solution[ind_T]
    
    p = Plots.plot(sol_h,label = "H")
    Plots.plot!(sol_v,label = "v")
    Plots.plot!(sol_m,label = "m")
    Plots.plot!(sol_T,label  = "T")
    Plots.display(p)

    ###Same ways to generate J
    x = res.solution
    y = res.multipliers
    n = NLPModels.get_nvar(nlp)
    m = NLPModels.get_ncon(nlp)
    nnzh = NLPModels.get_nnzh(nlp)
    Wi, Wj = NLPModels.hess_structure(nlp)
    Wz = zeros(nnzh)
    NLPModels.hess_coord!(nlp, x, y, Wz)
    W = sparse(Wi, Wj, Wz, n, n)
    nnzj = NLPModels.get_nnzj(nlp)
    Ji, Jj = NLPModels.jac_structure(nlp)
    Jz = zeros(nnzj)
    NLPModels.jac_coord!(nlp, x, Jz)
    J = sparse(Ji, Jj, Jz, m, n)
    J1 = Matrix(J)

    ###Add J_iu and calculate eigvalues and plot
    Xs = []
    Ys = []
    for i in range(1,length(tols))
        tol = tols[i]
        I_l[1,ind_h] = -(abs.(sol_h .- Lim_l_h).<= tol) .+ 0 
        I_l[2,ind_v] = -(abs.(sol_v .- Lim_l_v).<= tol) .+ 0 
        I_l[3,ind_m] = -(abs.(sol_m .- Lim_l_m).<= tol) .+ 0 
        I_l[4,ind_T] = -(abs.(sol_T .- Lim_l_T).<= tol) .+ 0 
        I_u[1,ind_m] = (abs.(sol_m .- Lim_u_m).<= tol) .+ 0
        I_u[2,ind_T] = (abs.(sol_T .- Lim_u_T).<= tol) .+ 0    
        I_lu = vcat(I_l,I_u)

        J_act = vcat(J1,I_lu)
        println("Added $(size(J_act)[1]-size(J1)[1]) lines to formal Jacobian as real inequal constrains")
        Z = nullspace(J_act)
        H = Z'*Symmetric(W, :L)*Z
        present_Heatmap(H,"H_act of euler explicit, tol = $(tol)")
        Eigs = eigvals(H)
        append!(Xs,Eigs)
        append!(Ys,[i for n in range(1,length(Eigs))])
    end
    Plots.scatter(Ys,Xs,xticks = (1:length(tols),tols),label = false,xlabel = "Tolerance",ylabel = "Eigvals",title = "Egivalues of Matrix H, by explicit Euler ")
    Plots.savefig("Variance in Eig with actual J, nh= $nh.png")
end

Calcul_eigvals_euler(100,[1e-8,1e-6,1e-5,1e-4,1e-2])
# Calcul_eigvals_models([rocket_model,rocket_model_hersim,rocket_model_euler_exp,rocket_model_euler_imp],["Trapezoidal","Hermite-Simpson","Euler-Explicite","Euler-Implicite"])
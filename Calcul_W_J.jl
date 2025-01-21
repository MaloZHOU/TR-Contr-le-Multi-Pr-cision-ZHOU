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

function Calcul_eigvals_act(nh = 100,tols = [1e-6])
    ###Generate the additional lines in J_act
    ###Paramaters
    model = rocket_model_hersim(nh)
    nlp = MathOptNLPModel(model)
    res = ipopt(nlp)
    
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
    ###Add J_iu and calculate eigvalues and plot
    Xs = []
    Ys = []
    for i in range(1,length(tols))
        tol = tols[i]
        Al = findall(res.multipliers_L .> tol)
        Au = findall(res.multipliers_U .> tol)
    
        nl = length(Al)
        nu = length(Au)
        Jact = sparse([Ji; m+1:m+nl; m+nl+1:m+nl+nu], [Jj; Al; Au], [Jz; ones(nl); ones(nu)], m+nl+nu, n)
        
        Z = nullspace(Matrix(Jact))
        H = Z'*Symmetric(W, :L)*Z
        present_Heatmap(H,"H_act/H_act of Herm-Sim, tol = $(tol)")
        Eigs = eigvals(H)
        append!(Xs,Eigs)
        append!(Ys,[i for n in range(1,length(Eigs))])
    end
    Plots.scatter(Ys,Xs,xticks = (1:length(tols),tols),label = false,xlabel = "Tolerance",ylabel = "Eigvals",title = "Egivalues of Matrix H, by Herm-Sim")
    Plots.savefig("H_act/Variance in Eig with actual J, Herm-Sim, nh= $nh.png")
end

Calcul_eigvals_act(100,[1e-8,1e-6,1e-5,1e-4,1e-2])
# Calcul_eigvals_models([rocket_model,rocket_model_hersim,rocket_model_euler_exp,rocket_model_euler_imp],["Trapezoidal","Hermite-Simpson","Euler-Explicite","Euler-Implicite"])
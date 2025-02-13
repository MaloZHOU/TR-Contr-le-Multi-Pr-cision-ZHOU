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
    return (mat .- min_val) ./ (max_val - min_val)
end

function present_Heatmap(Mat,name="")
    # norm_matrix = normalize_matrix(Mat)
    max_Mat = round(maximum(Mat),sigdigits = 4)
    min_Mat = round(minimum(Mat),sigdigits = 4)
    p = heatmap(norm_matrix, color=:grays, 
            xlabel="Columns", ylabel="Rows", title="Matrix Heatmap of H $name")
    #Plots.savefig(p,"$name,$Lims.png")
    return p
end

function Calcul_H_model(goddard_model,nh=100)
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
    # present_Heatmap(H,name_model)
    return H
end

nhs = [10,30,50,100,144,225,320,450,500]
D = (1 ./ nhs ) 
lgD = [log(i) for i in D]

function plot_Hmax_nhs(nhs,method)
    Maxs = zeros(length(nhs))
    eigs = zeros(length(nhs))
    for i in range(1,length(nhs))
        println(nhs[i])
        H_nh = Calcul_H_model(method,nhs[i])
        Max_nh = maximum(H_nh)
        Max_nh_eig = maximum(eigvals(H_nh))
        Maxs[i] = Max_nh
        eigs[i] = Max_nh_eig
    end
    return(Maxs,eigs)
end

function Plot_Compare(nhs,method,name)
    Cals = plot_Hmax_nhs(nhs,method)
    Maxs = Cals[1]
    eigs = Cals[2]
    lgM = [log(i) for i in Maxs]
    lgV = [log(i) for i in eigs]
    Plots.scatter(lgD,lgV,xlabel = "log(Delta)",ylabel="log(Max(eig(H)))"       #,ylim = [-10,-5],xlim = [-6.5,0]
                ,label = "$name, nh de 10 à 500",title = "Figure logarithme de valeur propre maximum")
    Plots.savefig("max(eig(H)),$name.png")
    Plots.scatter(lgD,lgM,label = "$name, nh de 10 à 500"                       #, xlim = [-7,-2],ylim = [-15,-5]
                ,xlabel="log(Delta)",ylabel = "log(Max(H))",title = "Figure logarithme de valeur maximum de H")
    Plots.savefig("max(H),$name.png")
    println("$name,Figures saved")
    return(Maxs,eigs)
end

Maxs,eigs = Plot_Compare(nhs,rocket_model,"Trapezoidal")

# Cals = plot_Hmax_nhs(nhs)
# Maxs = Cals[1]
# eigs = Cals[2]
# lgM = [log(i) for i in Maxs]
# lgV = [log(i) for i in eigs]

# Plots.scatter(lgD,lgV,ylim = [-10,-5],xlim = [-6.5,0],xlabel = "log(Delta)",ylabel="log(Max(eig(H)))",
#                 label = "Euler exp, nh de 10 à 500",title = "Figure logarithme de valeur propre maximum")
# Plots.savefig("max(eig(H)),Euler_explicite.png")
# Plots.scatter(lgD,lgM,label = "Euler exp, nh de 10 à 500",xlim = [-7,-2],ylim = [-15,-5]
#                 ,xlabel="log(Delta)",ylabel = "log(Max(H))",title = "Figure logarithme de valeur maximum de H")
# Plots.savefig("max(H),Euler_explicite.png")
# println("Figures saved")

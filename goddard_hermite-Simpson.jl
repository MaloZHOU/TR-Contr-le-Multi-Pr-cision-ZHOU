using JuMP
using Ipopt
#using COPSBenchmark
import Plots
using HSL_jll

# Goddard Rocket Problem
# Hermite-Simpson formulation
# Three hyperparameters to be checked before every run: 1) objective to optimize, at line 46,47; 2) tolerance applied to the model, at line 81
# 3) number of discritization nh, at line 96

function rocket_model_hersim(nh)
    h_0 = 1.0   #hauteur init
    v_0 = 0.0   #vitesse init
    m_0 = 1.0   #masse init
    g_0 = 1.0   #grativité init
    T_c = 3.5
    h_c = 500.0
    v_c = 620.0
    m_c = 0.6

    c = 0.5*sqrt(g_0 * h_0)
    m_f = m_c * m_0
    D_c = 0.5 * v_c * (m_0 / g_0)
    T_max = T_c * m_0 * g_0

    model = Model()

    @variables(model, begin
        1.0 <= h[i=0:nh,j=0:1],          (start=1.0)
        0.0 <= v[i=0:nh,j=0:1] <= Inf,          (start=i/nh*(1.0 - i/nh))
        m_f <= m[i=0:nh,j=0:1] <= m_0,   (start=(m_f - m_0)*(i/nh) + m_0)
        0.0 <= T[i=0:nh,j=0:1] <= T_max, (start=T_max/2.0)
        0.0 <= step,               (start=1/nh)
    end)

    @expressions(model, begin
        D[i=0:nh,j=0:1],  D_c*v[i,j]^2*exp(-h_c*(h[i,j] - h_0))/h_0
        g[i=0:nh,j=0:1],  g_0 * (h_0 / h[i,j])^2
        dh[i=0:nh,j=0:1], v[i,j]
        dv[i=0:nh,j=0:1], (T[i,j] - D[i,j] - m[i,j]*g[i,j]) / m[i,j]
        dm[i=0:nh,j=0:1], -T[i,j]/c
    end)
    
    #weight of each dimension of parrameters and target residuals 
    coef_obj = 0.7
    wh = h_0
    wv = wh/step
    wm = m_0
    sum_weight = wh+wv+wm
    wh = (1-coef_obj) * wh/sum_weight
    wv = (1-coef_obj) * wv/sum_weight
    wm = (1-coef_obj) * wm/sum_weight
    

    #Set Objective
    # @objective(model, Max, h[nh,0])
    # @objective(model, Max, h[nh,0])
    @objective(model, Min, (coef_obj-1)*h[nh,0] +sum( wh * (h[i+1,0] - h[i,0]-step*dh[i,0]) 
                                                    + wv * (v[i+1,0] - v[i,0]-step*dv[i,0]) 
                                                    + wm * (m[i+1,0] - m[i,0]-step*dm[i,0]) for i in 0:nh-1))
    
    #Hermite-Simpson Method
    @constraints(model,begin
        def_ref_h[i=1:nh-1], h[i,1] == 0.5 * (h[i,0] + h[i+1,0]) + 0.125 * step * (dh[i,0] - dh[i+1,0])
        def_ref_v[i=1:nh-1], v[i,1] == 0.5 * (v[i,0] + v[i+1,0]) + 0.125 * step * (dv[i,0] - dv[i+1,0])
        def_ref_m[i=1:nh-1], m[i,1] == 0.5 * (m[i,0] + m[i+1,0]) + 0.125 * step * (dm[i,0] - dm[i+1,0])

        con_dh[i=1:nh], h[i,0] == h[i-1,0] + 1/6 * step * (dh[i-1,0] + dh[i,0] + 4 * dh[i-1,1])
        con_dv[i=1:nh], v[i,0] == v[i-1,0] + 1/6 * step * (dv[i-1,0] + dv[i,0] + 4 * dv[i-1,1])
        con_dm[i=1:nh], m[i,0] == m[i-1,0] + 1/6 * step * (dm[i-1,0] + dm[i,0] + 4 * dm[i-1,1])
    end)
    #Boundary constraints
    @constraints(model, begin
        h_ic, h[0,0] == h_0
        v_ic, v[0,0] == v_0
        m_ic, m[0,0] == m_0
        m_fc, m[nh,0] == m_f

        h_ic_1, h[0,1] == h_0
        v_ic_1, v[0,1] == v_0
        m_ic_1, m[0,1] == m_0
    end)

    return model
end

function Generate_thrust_hersim(nhs=nhs)
    P = Plots.plot(xlabel="Temps", ylabel="Value")
    Thrusts = [[[] for i in range(1,length(nhs))] for j in range(1,2)]
    for i in range(1,length(nhs))
        nh = nhs[i]
        println(nh)
        model = rocket_model_hersim(nh)
        JuMP.set_optimizer(model, Ipopt.Optimizer)
        JuMP.set_attribute(model,"tol",1e-8)
        JuMP.set_attribute(model, "hsllib", HSL_jll.libhsl_path)
        JuMP.set_attribute(model, "linear_solver", "ma57")
        JuMP.optimize!(model)
        T_value = value.(model[:T]);
        T_Array = Array(T_value[:,0]);
        T_Array_dua = Array(T_value[:,1]);
        Thrusts[1][i] = T_Array
        Thrusts[2][i] = T_Array_dua
        Plots.plot!(LinRange(0,0.2,length(T_Array)),T_Array,label="T values for nh = $nh")
    end
    Plots.display(P)
    return P
end

function uni_plot(Ls,names)
    p = Plots.plot()
    for j in range(1,length(Ls))
        Data = [ (i-minimum(Ls[j]))/(maximum(Ls[j])-minimum(Ls[j])) for i in Ls[j]]
        p = Plots.plot!(LinRange(0,1,length(Data)),Data,label =names[j])
    end
    return p
end

# nhs = [100,500,1000,5000,10000]
# Generate_thrust_hersim(nhs)


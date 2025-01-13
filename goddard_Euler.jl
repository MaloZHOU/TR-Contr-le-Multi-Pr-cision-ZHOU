using JuMP
using Ipopt
#using COPSBenchmark
import Plots
using HSL_jll

# Goddard Rocket Problem
# Explicite and implicite methode of Euler: 
# dxn = f(xm) and dxn = f(xn+1)

function rocket_model_euler_exp(nh)
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
        1.0 <= h[i=0:nh],          (start=1.0)
        0.0 <= v[i=0:nh] <= Inf,   (start=i/nh*(1.0 - i/nh))
        m_f <= m[i=0:nh] <= m_0,   (start=(m_f - m_0)*(i/nh) + m_0)
        0.0 <= T[i=0:nh] <= T_max, (start=T_max/2.0)
        0.0 <= step,                   (start=1/nh)
    end)

    @expressions(model, begin
        D[i=0:nh],  D_c*v[i]^2*exp(-h_c*(h[i] - h_0))/h_0
        g[i=0:nh],  g_0 * (h_0 / h[i])^2
        dh[i=0:nh], v[i]
        dv[i=0:nh], (T[i] - D[i] - m[i]*g[i]) / m[i]
        dm[i=0:nh], -T[i]/c
    end)
    
    #Set Objective
    # @objective(model, Max, h[nh,0])
    @objective(model, Max, h[nh])
    
    #Hermite-Simpson Method
    @constraints(model,begin
        euler_exp_h[i=0:nh-1], h[i+1] == h[i] + dh[i]
        euler_exp_v[i=0:nh-1], v[i+1] == v[i] + dv[i]
        euler_exp_m[i=0:nh-1], m[i+1] == m[i] + dm[i]
    end)
    #Boundary constraints
    @constraints(model, begin
        h_ic, h[0] == h_0
        v_ic, v[0] == v_0
        m_ic, m[0] == m_0
        m_fc, m[nh] == m_f
    end)

    return model
end

function rocket_model_euler_imp(nh)
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
        1.0 <= h[i=0:nh],          (start=1.0)
        0.0 <= v[i=0:nh] <= Inf,   (start=i/nh*(1.0 - i/nh))
        m_f <= m[i=0:nh] <= m_0,   (start=(m_f - m_0)*(i/nh) + m_0)
        0.0 <= T[i=0:nh] <= T_max, (start=T_max/2.0)
        0.0 <= step,                   (start=1/nh)
    end)

    @expressions(model, begin
        D[i=0:nh],  D_c*v[i]^2*exp(-h_c*(h[i] - h_0))/h_0
        g[i=0:nh],  g_0 * (h_0 / h[i])^2
        dh[i=0:nh], v[i]
        dv[i=0:nh], (T[i] - D[i] - m[i]*g[i]) / m[i]
        dm[i=0:nh], -T[i]/c
    end)
    
    #Set Objective
    # @objective(model, Max, h[nh,0])
    @objective(model, Max, h[nh])
    
    #Hermite-Simpson Method
    @constraints(model,begin
        euler_exp_h[i=0:nh-1], h[i+1] == h[i] + dh[i+1]
        euler_exp_v[i=0:nh-1], v[i+1] == v[i] + dv[i+1]
        euler_exp_m[i=0:nh-1], m[i+1] == m[i] + dm[i+1]
    end)
    #Boundary constraints
    @constraints(model, begin
        h_ic, h[0] == h_0
        v_ic, v[0] == v_0
        m_ic, m[0] == m_0
        m_fc, m[nh] == m_f
    end)

    return model
end

function Generate_thrust_exp(nhs=nhs)
    P = Plots.plot(xlabel="Temps", ylabel="Value")
    for i in range(1,length(nhs))
        nh = nhs[i]
        println(nh)
        model = rocket_model_euler_exp(nh)
        JuMP.set_optimizer(model, Ipopt.Optimizer)
        JuMP.set_attribute(model,"tol",1e-9)
        JuMP.set_attribute(model, "hsllib", HSL_jll.libhsl_path)
        JuMP.set_attribute(model, "linear_solver", "ma57")
        JuMP.optimize!(model)
        T_value = value.(model[:T]);
        T_Array = Array(T_value[:]);
        Plots.plot!(LinRange(0,0.2,length(T_Array)),T_Array,label="T values for nh = $nh")
    end
    Plots.display(P)
    return P
end

function Generate_thrust_imp(nhs=nhs)
    P = Plots.plot(xlabel="Temps", ylabel="Value")
    for i in range(1,length(nhs))
        nh = nhs[i]
        println(nh)
        model = rocket_model_euler_imp(nh)
        JuMP.set_optimizer(model, Ipopt.Optimizer)
        JuMP.set_attribute(model,"tol",1e-9)
        JuMP.set_attribute(model, "hsllib", HSL_jll.libhsl_path)
        JuMP.set_attribute(model, "linear_solver", "ma57")
        JuMP.optimize!(model)
        T_value = value.(model[:T]);
        T_Array = Array(T_value[:]);
        Plots.plot!(LinRange(0,0.2,length(T_Array)),T_Array,label="T values for nh = $nh")
    end
    Plots.display(P)
    return P
end

nhs = [50,100,200,500,1000]
p = Generate_thrust_exp(nhs)
Plots.savefig(p,"exp.png")
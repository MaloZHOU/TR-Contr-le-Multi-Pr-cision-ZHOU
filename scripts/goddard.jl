
using JuMP
using Ipopt
#using COPSBenchmark
import Plots
using HSL_jll

# Goddard Rocket Problem
# Trapezoidal formulation

function rocket_model(nh)
    h_0 = 1.0
    v_0 = 0.0
    m_0 = 1.0
    g_0 = 1.0
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
        0.0 <= step,               (start=1/nh)
    end)

    @expressions(model, begin
        D[i=0:nh],  D_c*v[i]^2*exp(-h_c*(h[i] - h_0))/h_0
        g[i=0:nh],  g_0 * (h_0 / h[i])^2
        dh[i=0:nh], v[i]
        dv[i=0:nh], (T[i] - D[i] - m[i]*g[i]) / m[i]
        dm[i=0:nh], -T[i]/c
    end)

    @objective(model, Max, h[nh])
    # Dynamics
    @constraints(model, begin
        con_dh[i=1:nh], h[i] == h[i-1] + 0.5 * step * (dh[i] + dh[i-1])
        con_dv[i=1:nh], v[i] == v[i-1] + 0.5 * step * (dv[i] + dv[i-1])
        con_dm[i=1:nh], m[i] == m[i-1] + 0.5 * step * (dm[i] + dm[i-1])
    end)

    # Boundary constraints
    @constraints(model, begin
        h_ic, h[0] == h_0
        v_ic, v[0] == v_0
        m_ic, m[0] == m_0
        m_fc, m[nh] == m_f
    end)

    return model
end

function rocket_model_res(nh,coef_obj = 1.0)
    h_0 = 1.0
    v_0 = 0.0
    m_0 = 1.0
    g_0 = 1.0
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
        h[i=0:nh],          (start=1.0)
        v[i=0:nh],   (start=i/nh*(1.0 - i/nh))
        m[i=0:nh],   (start=(m_f - m_0)*(i/nh) + m_0)
        T[i=0:nh], (start=T_max/2.0)
        0.0 <= step,               (start=1/nh)
    end)

    @expressions(model, begin
        D[i=0:nh],  D_c*v[i]^2*exp(-h_c*(h[i] - h_0))/h_0
        g[i=0:nh],  g_0 * (h_0 / h[i])^2
        dh[i=0:nh], v[i]
        dv[i=0:nh], (T[i] - D[i] - m[i]*g[i]) / m[i]
        dm[i=0:nh], -T[i]/c
    end)
    #weight of each dimension of parrameters and target residuals 
    wh = h_0
    wv = wh/step
    wm = m_0
    sum_weight = wh+wv+wm
    wh = (1-coef_obj) * wh/sum_weight
    wv = (1-coef_obj) * wv/sum_weight
    wm = (1-coef_obj) * wm/sum_weight
    
    #Set Objective
    # @objective(model, Max, h[nh])
    @objective(model, Min,(-coef_obj)*h[nh] +sum(wh * (h[i+1] - h[i]-step*dh[i])^2 
                                                    + wv * (v[i+1] - v[i]-step*dv[i])^2 
                                                    + wm * (m[i+1] - m[i]-step*dm[i])^2 for i in 0:nh-1))
    
    # Dynamics
    @constraints(model, begin
        con_dh[i=1:nh], h[i] == h[i-1] + 0.5 * step * (dh[i] + dh[i-1])
        con_dv[i=1:nh], v[i] == v[i-1] + 0.5 * step * (dv[i] + dv[i-1])
        con_dm[i=1:nh], m[i] == m[i-1] + 0.5 * step * (dm[i] + dm[i-1])
    end)

    
    @constraints(model, begin
        ineq_h_d[i=0:nh], h[i] >= 1 
        ineq_v_d[i=0:nh], v[i] >= 0.0 
        ineq_m_d[i=0:nh], m[i] >= m_f 
        ineq_T_d[i=0:nh], T[i] >= 0.0
        ineq_m_u[i=0:nh], m[i] <= m_0 
        ineq_T_u[i=0:nh], T[i] <= T_max
    end)

    # Boundary constraints
    @constraints(model, begin
        h_ic, h[0] == h_0
        v_ic, v[0] == v_0
        m_ic, m[0] == m_0
        m_fc, m[nh] == m_f
    end)

    return model
end

# Solve problem with Ipopt
function Generate_thrust(nhs=nhs)
    Thrusts = [[] for i in range(1,length(nhs))]
    P = Plots.plot(xlabel="Time", ylabel="Thrust Value",title="tol=1e-8, Trapezoidal")
    for i in range(1,length(nhs))
        println(nhs[i])
        nh = nhs[i]
        model = rocket_model(nh)
        JuMP.set_optimizer(model, Ipopt.Optimizer)
        JuMP.set_attribute(model,"tol",1e-8)
        JuMP.set_attribute(model, "hsllib", HSL_jll.libhsl_path)
        JuMP.set_attribute(model, "linear_solver", "ma57")
        JuMP.optimize!(model)
        T_value = value.(model[:T]);
        T_Array = Array(T_value);
        Thrusts[i] = T_Array
        Plots.plot!(LinRange(0,0.2,length(T_Array)),T_Array,label="u values for nh = $nh")
    end
    print("Loop done")
    return P
end

function Generate_thrust_obj(nh,coef_objs)
    P = Plots.plot(xlabel="Time", ylabel="Thrust Value",title = "nh=$nh, Trapezoidal")
    for i in range(1,length(coef_objs))
        println(coef_objs[i])
        coef_obj = coef_objs[i]
        model = rocket_model_res(nh,coef_obj)
        JuMP.set_optimizer(model, Ipopt.Optimizer)
        JuMP.set_attribute(model, "hsllib", HSL_jll.libhsl_path)
        JuMP.set_attribute(model, "linear_solver", "ma57")
        JuMP.optimize!(model)
        T_value = value.(model[:T]);
        T_Array = Array(T_value);
        Plots.plot!(LinRange(0,0.2,length(T_Array)),T_Array,label="u values for Wobj=$coef_obj")
    end
    print("Loop done")
    return P
end

function Generate_thrust_tols(tols,nh=500)
    P = Plots.plot(xlabel="Time", ylabel="Thrust value",title="nh = $nh, Trapezoidal")
    for i in range(1,length(tols))
        tol = tols[i]
        model = rocket_model(nh)
        JuMP.set_optimizer(model, Ipopt.Optimizer)
        JuMP.set_attribute(model,"tol",tol)
        JuMP.set_attribute(model, "hsllib", HSL_jll.libhsl_path)
        JuMP.set_attribute(model, "linear_solver", "ma57")
        JuMP.optimize!(model)
        T_value = value.(model[:T]);
        T_Array = Array(T_value);
        Plots.plot!(LinRange(0,0.2,length(T_Array)),T_Array,label="u values for tol = $tol")
    end
    print("Loop done")
    return P
end

# P = Generate_thrust([50,100,500,1000,2000])
# Plots.display(P)
# Plots.savefig("Nouvel_obj_trapèze_tol=1e-8.png")
using JuMP
using Ipopt
import Plots
using HSL_jll

function rocket_model_trap(nh,ep)
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

    #Set Objective
    @objective(model, Max, h[nh] + ep * sum(T[i]^2 for i in 0:nh))
    
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

function rocket_model_hersim(nh,ep)
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
    
    #Set Objective
    # @objective(model, Max, h[nh,0])
    @objective(model, Max, h[nh,0] + ep * sum(T[i,0]^2 for i in 0:nh))
    
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


function Generate_Trap_nhs_tol_ep(nhs,tol,ep)
    p = Plots.plot(title = "tol = $tol, Trapezoidal",xlabel="Time", ylabel="Thrust Value")
    for nh in nhs
        model = rocket_model_trap(nh,ep)
        JuMP.set_optimizer(model, Ipopt.Optimizer)
        JuMP.set_attribute(model, "tol", tol)
        JuMP.set_attribute(model, "hsllib", HSL_jll.libhsl_path)
        JuMP.set_attribute(model, "linear_solver", "ma57")
        JuMP.optimize!(model)
        Thrust = collect(JuMP.value.(model[:T]))
        Plots.plot!(LinRange(0,1,length(Thrust)),Thrust,label="u values for nh = $nh")
    end
    Plots.display(p)
    return p
end

function Generate_hersim_nhs_tol_ep(nhs,tol,ep)
    p = Plots.plot(title = "tol = $tol, epsilon = $ep, Her-Sim",xlabel="Temps", ylabel="Value")
    for nh in nhs
        model = rocket_model_hersim(nh,ep)
        JuMP.set_optimizer(model, Ipopt.Optimizer)
        JuMP.set_attribute(model, "tol", tol)
        JuMP.set_attribute(model, "hsllib", HSL_jll.libhsl_path)
        JuMP.set_attribute(model, "linear_solver", "ma57")
        JuMP.optimize!(model)
        Thrust = collect(JuMP.value.(model[:T]))[:, 1]
        Plots.plot!(LinRange(0,1,length(Thrust)),Thrust,label="T values for nh = $nh")
    end
    Plots.display(p)
    return p
end

nhs = [50,100,200,500,1000,5000,10000]
tols = [1e-6,1e-7,1e-8,1e-10,1e-12]
eps = [1e-14]

for tol in tols
    p = Generate_Trap_nhs_tol_ep(nhs,tol,eps[1])
    Plots.savefig("Comparaison_nhs_tol_ep/Trapezoidal,tol = $tol.png")
    println("tol = $tol, saved")
end

# for tol in tols
#     for ep in eps
#         p = Generate_hersim_nhs_tol_ep(nhs,tol,ep)
#         Plots.savefig("Photo/Comparaison_nhs_tol_ep/Hermite_simpson,tol = $tol, epsilon = $ep.png")
#         println("tol = $tol, epsilon = $ep saved")
#     end
# end

# println("loop end")
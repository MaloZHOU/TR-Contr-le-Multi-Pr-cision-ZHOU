using JuMP
using Ipopt
#using COPSBenchmark
import Plots
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
        0.0 <= v[i=0:nh],          (start=i/nh*(1.0 - i/nh))
        m_f <= m[i=0:nh] <= m_0,   (start=(m_f - m_0)*(i/nh) + m_0)
        0.0 <= T[i=0:nh] <= T_max, (start=T_max/2.0)
        
        # 0.0 <= step,               (start=1/nh)
        1e-8 <= step[i=1:nh],      (start=1/nh)
        # 0.1/nh <= step[i=1:nh] <= 10/nh,      (start=1/nh)
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
        con_dh[i=1:nh], h[i] == h[i-1] + 0.5 * step[i] * (dh[i] + dh[i-1])
        con_dv[i=1:nh], v[i] == v[i-1] + 0.5 * step[i] * (dv[i] + dv[i-1])
        con_dm[i=1:nh], m[i] == m[i-1] + 0.5 * step[i] * (dm[i] + dm[i-1])
        
        ## Set bounded steps
        con_step_bond[i=1:nh], 0.1/nh <= step[i] <= 10/nh
        
        ## Constraints on total time
        con_step, sum(step) == 1
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

function Generate_thrust(nhs=nhs)
    Thrusts = [[] for i in range(1,length(nhs))]
    Steps = [[] for i in range(1,length(nhs))]
    
    P = Plots.plot(xlabel="Temps", ylabel="Value")
    for i in range(1,length(nhs))
        nh = nhs[i]
        model = rocket_model(nh)
        JuMP.set_optimizer(model, Ipopt.Optimizer)
        JuMP.optimize!(model)
        
        Thrusts[i] = Array(value.(model[:T]))
        Steps[i] = [0;cumsum(Array(value.(model[:step])))]
        p1 = Plots.plot([0;cumsum(Steps)],title = "step by indice",ylabel = "time",label = "added steps for nh = $nh");
        p2 = Plots.plot(Thrusts,title = "T by indice",ylabel = "value",label = "T value for nh = $nh");
        p3 = Plots.plot([0;cumsum(Steps)],Thrusts,title = "T by time",ylabel = "value",label = "T value for nh = $nh",xlabel = "Time")
        Plots.plot!(p1,p2,p3,layout = (1,3))

    end
    print("Loop done")
    return P
end

p = Generate_thrust([5000]);
Plots.display(p)
Plots.savefig("Pas_libre.png")
println("showen")

# nhs = [100,500,1000,5000,10000] 

# function main(nhs = nhs)
#     Plot_Thrust = Generate_thrust(nhs)
#     Plots.display(Plot_Thrust)
#     readline()
# end

# main()

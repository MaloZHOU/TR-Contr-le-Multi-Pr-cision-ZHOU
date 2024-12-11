using JuMP
using Ipopt
#using COPSBenchmark
import Plots

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
        0.0 <= v[i=0:nh,j=0:1],          (start=i/nh*(1.0 - i/nh))
        m_f <= m[i=0:nh,j=0:1] <= m_0,   (start=(m_f - m_0)*(i/nh) + m_0)
        0.0 <= T[i=0:nh] <= T_max, (start=T_max/2.0)
        0.0 <= step,               (start=1/nh)
    end)

    @expressions(model, begin
        D[i=0:nh,j=0:1],  D_c*v[i,j]^2*exp(-h_c*(h[i,j] - h_0))/h_0
        g[i=0:nh,j=0:1],  g_0 * (h_0 / h[i,j])^2
        dh[i=0:nh,j=0:1], v[i,j]
        dv[i=0:nh,j=0:1], (T[i] - D[i,j] - m[i,j]*g[i,j]) / m[i,j]
        dm[i=0:nh,j=0:1], -T[i]/c
    end)

    @objective(model, Max, h[nh,0])
    #Hermite-Simpson Method
    @constraints(model,begin
        def_ref_h[i=1:nh-1], h[i,1] == 0.5 * (h[i,0] + h[i+1,0]) + 0.125 * step * (dh[i,0] - dh[i+1,0])
        def_ref_v[i=1:nh-1], v[i,1] == 0.5 * (v[i,0] + v[i+1,0]) + 0.125 * step * (dv[i,0] - dv[i+1,0])
        def_ref_m[i=1:nh-1], m[i,1] == 0.5 * (m[i,0] + m[i+1,0]) + 0.125 * step * (dm[i,0] - dm[i+1,0])

        con_dh[i=1:nh], h[i,0] == h[i-1,0] + 1/6 * step * (dh[i-1,0] + dh[i,0] + 4 * dh[i,1])
        con_dv[i=1:nh], v[i,0] == v[i-1,0] + 1/6 * step * (dv[i-1,0] + dv[i,0] + 4 * dv[i,1])
        con_dm[i=1:nh], m[i,0] == m[i-1,0] + 1/6 * step * (dm[i-1,0] + dm[i,0] + 4 * dm[i,1])
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
        model = rocket_model_hersim(nh)
        JuMP.set_optimizer(model, Ipopt.Optimizer)
        JuMP.optimize!(model)
        T_value = value.(model[:T]);
        print(T_value)
        T_Array = Array(T_value[:,0]);
#        T_Array_dua = Array(T_value[:,1]);
        Thrusts[1][i] = T_Array
#        Thrusts[2][i] = T_Array_dua
        Plots.plot!(LinRange(0,0.2,length(T_Array)),T_Array,label="T values for nh = $nh")
    end
    Plots.display(P)
    return Thrusts
end

MModel = rocket_model_hersim(100)
JuMP.set_optimizer(MModel, Ipopt.Optimizer)
JuMP.optimize!(MModel)
T_value = value.(MModel[:T]);
print(T_value)
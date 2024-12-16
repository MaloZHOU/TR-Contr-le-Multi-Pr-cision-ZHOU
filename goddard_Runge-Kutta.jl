using JuMP
using Ipopt
#using COPSBenchmark
import Plots

function rocket_model_Runge(nh)
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
        1.0 <= h[i=0:nh,j=0:4],          (start=1.0)
        0.0 <= v[i=0:nh,j=0:4],          (start=i/nh*(1.0 - i/nh))
        m_f <= m[i=0:nh,j=0:4] <= m_0,   (start=(m_f - m_0)*(i/nh) + m_0)
        0.0 <= T[i=0:nh,j=0:4] <= T_max, (start=T_max/2.0)
        0.0 <= step,               (start=1/nh)
    end)

    @expressions(model, begin
        D[i=0:nh,j=0:4],  D_c*v[i,j]^2*exp(-h_c*(h[i,j] - h_0))/h_0
        g[i=0:nh,j=0:4],  g_0 * (h_0 / h[i,j])^2
        dh[i=0:nh,j=0:4], v[i,j]
        dv[i=0:nh,j=0:4], (T[i,j] - D[i,j] - m[i,j]*g[i,j]) / m[i,j]
        dm[i=0:nh,j=0:4], -T[i,j]/c
    end)

    @objective(model, Max, h[nh,0])

    #Runge-Kutta Method
    @constraints(model,begin
    #x1 = x0 + 0.5 * k1, x in {h,v,m}:
        def_h1[i=1:nh], h[i,1] == 0.5 * step * dh[i,0] + h[i,0]
        def_v1[i=1:nh], v[i,1] == 0.5 * step * dv[i,0] + v[i,0]
        def_m1[i=1:nh], m[i,1] == 0.5 * step * dm[i,0] + m[i,0]

    #x2 = x0 + 0.5 * k2, x in {h,v,m}:
        def_h2[i=1:nh], h[i,2] == 0.5 * step * dh[i,1] + h[i,0]
        con_v2[i=1:nh], v[i,2] == 0.5 * step * dv[i,1] + v[i,0]
        con_m2[i=1:nh], m[i,2] == 0.5 * step * dm[i,1] + m[i,0]

    #x3 = x0 + k3, x in {h,v,m}:
        def_h3[i=1:nh], h[i,3] == step * dh[i,2] + h[i,0]
        con_v3[i=1:nh], v[i,3] == step * dv[i,2] + v[i,0]
        con_m3[i=1:nh], m[i,3] == step * dm[i,2] + m[i,0]       

    #x4 = k4
        def_h4[i=1:nh], h[i,4] == step * dh[i,3]
        con_v4[i=1:nh], v[i,4] == step * dv[i,3]
        con_m4[i=1:nh], m[i,4] == step * dm[i,3]

    #x0[i+1] = x0[i] + (1 / 6) * (k1 + 2*k2 + 2*k3 + k4)
        runge_kutta_h[i=0:nh-1], h[i+1,0] == h[i,0] + (1/6) * ( 2 * (h[i,1]-h[i,0]) + 4 * (h[i,2]-h[i,0]) + 2 * (h[i,3]-h[i,0]) +h[i,4])
        runge_kutta_v[i=0:nh-1], v[i+1,0] == v[i,0] + (1/6) * ( 2 * (v[i,1]-v[i,0]) + 4 * (v[i,2]-v[i,0]) + 2 * (v[i,3]-v[i,0]) +v[i,4])
        runge_kutta_m[i=0:nh-1], m[i+1,0] == m[i,0] + (1/6) * ( 2 * (m[i,1]-m[i,0]) + 4 * (m[i,2]-m[i,0]) + 2 * (m[i,3]-m[i,0]) +m[i,4])
    end)
    
    #Boundary constraints
    @constraints(model, begin
        h_ic[j=0:4], h[0,j] == h_0
        v_ic[j=0:4], v[0,j] == v_0
        m_ic[j=0:4], m[0,j] == m_0
        m_fc, m[nh,0] == m_f
    end)

    return model
end

function Generate_thrust_Runge(nhs=nhs)
    P = Plots.plot(xlabel="Temps", ylabel="Value")
    Thrusts = [[] for i in range(1,length(nhs))]
    for i in range(1,length(nhs))
        nh = nhs[i]
        model = rocket_model_Runge(nh)
        JuMP.set_optimizer(model, Ipopt.Optimizer)
        JuMP.optimize!(model)
        T_value = value.(model[:T]);
        T_Array = Array(T_value[:,0]);
        Thrusts[i] = T_Array
        Plots.plot!(LinRange(0,1,length(T_Array)),T_Array,label="T values for nh = $nh")
    end
    Plots.display(P)
    return Thrusts
end

nh = 50
MModel = rocket_model_Runge(nh)
JuMP.set_optimizer(MModel, Ipopt.Optimizer)
JuMP.optimize!(MModel)
T_value = value.(MModel[:T]);
T_Array = Array(T_value[:,0]);
Plots.plot(LinRange(0,1,length(T_Array)),T_Array,label="T values for nh = $nh",xlabel = "Temps", ylabel = "T value", title = "Méthode de Runge_Kutta")
Plots.savefig("Runge-Kutta, nh = $nh.png")
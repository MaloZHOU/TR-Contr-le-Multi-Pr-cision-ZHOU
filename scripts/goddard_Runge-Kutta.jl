using JuMP
using Ipopt
#using COPSBenchmark
import Plots
using HSL_jll


# k_1 = h_i f_i\\
# k_2 = h_1 f(y_i+\frac{1}{2}k_1,t_i+\frac{1}{2}h_i) \\
# k_3 = h_i f(y_i+\frac{1}{2}k_2,t_i+\frac{1}{2}h_i) \\ 
# k_4 = h_1 f(y_i+k_3,t_{i+1}) \\
# y_{i+1} = y_i+ \frac{1}{6}(k_1+2k_2+2k_3+k_4)

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
        0.0 <= v[i=0:nh,j=0:4]<= Inf,          (start=i/nh*(1.0 - i/nh))
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
        def_h1[i=0:nh], h[i,1] == 0.5 * step * dh[i,0] + h[i,0]
        def_v1[i=0:nh], v[i,1] == 0.5 * step * dv[i,0] + v[i,0]
        def_m1[i=0:nh], m[i,1] == 0.5 * step * dm[i,0] + m[i,0]

    #x2 = x0 + 0.5 * k2, x in {h,v,m}:
        def_h2[i=0:nh], h[i,2] == 0.5 * step * dh[i,1] + h[i,0]
        con_v2[i=0:nh], v[i,2] == 0.5 * step * dv[i,1] + v[i,0]
        con_m2[i=0:nh], m[i,2] == 0.5 * step * dm[i,1] + m[i,0]

    #x3 = x0 + k3, x in {h,v,m}:
        def_h3[i=0:nh], h[i,3] == step * dh[i,2] + h[i,0]
        con_v3[i=0:nh], v[i,3] == step * dv[i,2] + v[i,0]
        con_m3[i=0:nh], m[i,3] == step * dm[i,2] + m[i,0]       

    #x4 = k4
        def_h4[i=0:nh], h[i,4] == step * dh[i,3]
        con_v4[i=0:nh], v[i,4] == step * dv[i,3]
        con_m4[i=0:nh], m[i,4] == step * dm[i,3]

    #x0[i+1] = x0[i] + (1 / 6) * (k1 + 2*k2 + 2*k3 + k4)
        runge_kutta_h[i=1:nh], h[i,0] == h[i-1,0] + (1/6) * ( 2 * (h[i-1,1]-h[i-1,0]) + 4 * (h[i-1,2]-h[i-1,0]) + 2 * (h[i-1,3]-h[i-1,0]) +h[i-1,4])
        runge_kutta_v[i=1:nh], v[i,0] == v[i-1,0] + (1/6) * ( 2 * (v[i-1,1]-v[i-1,0]) + 4 * (v[i-1,2]-v[i-1,0]) + 2 * (v[i-1,3]-v[i-1,0]) +v[i-1,4])
        runge_kutta_m[i=1:nh], m[i,0] == m[i-1,0] + (1/6) * ( 2 * (m[i-1,1]-m[i-1,0]) + 4 * (m[i-1,2]-m[i-1,0]) + 2 * (m[i-1,3]-m[i-1,0]) +m[i-1,4])
    end)
    
    #Boundary constraints
    @constraints(model, begin
        h_ic, h[0,0] == h_0
        v_ic, v[0,0] == v_0
        m_ic, m[0,0] == m_0
        m_fc0, m[nh,0] == m_f   
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

function uni_plot(Ls,names)
    p = Plots.plot()
    for j in range(1,length(Ls))
        Data = [ (i-minimum(Ls[j]))/(maximum(Ls[j])-minimum(Ls[j])) for i in Ls[j]]
        p = Plots.plot!(LinRange(0,1,length(Data)),Data,label =names[j])
    end
    return p
end

# p = uni_plot([1,2,3],"test")
# Plots.display(p)
# Plots.savefig(p,"test.png")
nh = 100
MModel = rocket_model_Runge(nh)
JuMP.set_attribute(MModel, "tol", 1e-10)
JuMP.set_attribute(MModel, "hsllib", HSL_jll.libhsl_path)
JuMP.set_attribute(MModel, "linear_solver", "ma57")
JuMP.set_optimizer(MModel, Ipopt.Optimizer)
JuMP.optimize!(MModel)
T_value = value.(MModel[:T]);
T_Array = Array(T_value[:,0]);

h_value = value.(MModel[:h]);
h_Array = Array(h_value[:,0]);

m_value = value.(MModel[:m]);
m_Array = Array(m_value[:,0]);

println(T_Array[2])
p = uni_plot([T_Array,h_Array,m_Array],["T","h","m"])
Plots.savefig(p,"Runge-Kutta, nh = $nh.png")
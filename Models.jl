##This .jl file aims to generate Goddard problem solutions with four possible dimensions to modify:
##nh, tol, coef_obj and scheme of discritization.
##To extend this function with more dimension of tuning, please change the Para_list defined at line 230, defination of conf at line 239 and the loop from 255 to 272

using JuMP
using Ipopt
import Plots
using HSL_jll
using IterTools

h_0 = 1.0                       #Initial height
v_0 = 0.0                       #Initial velocity
m_0 = 1.0                       #Initial mass
g_0 = 1.0                       #Initial grativity
T_c = 3.5   
h_c = 500.0
v_c = 620.0
m_c = 0.6                       
c = 0.5*sqrt(g_0 * h_0)         #power coefficient
m_f = m_c * m_0                 #Final mass
D_c = 0.5 * v_c * (m_0 / g_0)   #Drag coefficient
T_max = T_c * m_0 * g_0         #Maximum Thrust

function rocket_model_euler_exp(nh,coef_obj)
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
    #weight of each dimension of parameters and target residuals 
    wh = h_0
    wv = wh/step
    wm = m_0
    sum_weight = wh+wv+wm
    wh = (1-coef_obj) * wh/sum_weight
    wv = (1-coef_obj) * wv/sum_weight
    wm = (1-coef_obj) * wm/sum_weight

    @objective(model, Min,(-coef_obj)*h[nh] +sum(wh * (h[i+1] - h[i]-step*dh[i])^2 
                                                    + wv * (v[i+1] - v[i]-step*dv[i])^2 
                                                    + wm * (m[i+1] - m[i]-step*dm[i])^2 for i in 0:nh-1))    

    #Boundary constraints
    @constraints(model, begin
        h_ic, h[0] == h_0
        v_ic, v[0] == v_0
        m_ic, m[0] == m_0
        m_fc, m[nh] == m_f
    end)

    #explicit Euler
    @constraints(model,begin
    euler_exp_h[i=0:nh-1], h[i+1] == h[i] + step *dh[i]
    euler_exp_v[i=0:nh-1], v[i+1] == v[i] + step *dv[i]
    euler_exp_m[i=0:nh-1], m[i+1] == m[i] + step *dm[i]
    end)

    return model
end

function rocket_model_euler_imp(nh,coef_obj)
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
    #weight of each dimension of parameters and target residuals 
    wh = h_0
    wv = wh/step
    wm = m_0
    sum_weight = wh+wv+wm
    wh = (1-coef_obj) * wh/sum_weight
    wv = (1-coef_obj) * wv/sum_weight
    wm = (1-coef_obj) * wm/sum_weight
    @objective(model, Min,(-coef_obj)*h[nh] +sum(wh * (h[i+1] - h[i]-step*dh[i])^2 
                                                    + wv * (v[i+1] - v[i]-step*dv[i])^2 
                                                    + wm * (m[i+1] - m[i]-step*dm[i])^2 for i in 0:nh-1))
    
    #Boundary constraints
    @constraints(model, begin
        h_ic, h[0] == h_0
        v_ic, v[0] == v_0
        m_ic, m[0] == m_0
        m_fc, m[nh] == m_f
    end)

    #implicit Euler Method
    @constraints(model,begin
        euler_exp_h[i=0:nh-1], h[i+1] == h[i] + step * dh[i+1]
        euler_exp_v[i=0:nh-1], v[i+1] == v[i] + step * dv[i+1]
        euler_exp_m[i=0:nh-1], m[i+1] == m[i] + step * dm[i+1]
    end)

    return model
end

function rocket_model_Traper(nh,coef_obj)
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
    #weight of each dimension of parameters and target residuals 
    wh = h_0
    wv = wh/step
    wm = m_0
    sum_weight = wh+wv+wm
    wh = (1-coef_obj) * wh/sum_weight
    wv = (1-coef_obj) * wv/sum_weight
    wm = (1-coef_obj) * wm/sum_weight
    @objective(model, Min, (-coef_obj)*h[nh] +sum( wh * (h[i+1] - h[i]-step*dh[i])^2 
                                                    + wv * (v[i+1] - v[i]-step*dv[i])^2 
                                                    + wm * (m[i+1] - m[i]-step*dm[i])^2 for i in 0:nh-1)) 

    # Boundary constraints
    @constraints(model, begin
        h_ic, h[0] == h_0
        v_ic, v[0] == v_0
        m_ic, m[0] == m_0
        m_fc, m[nh] == m_f
    end)

    # Trapezoidal Method
    @constraints(model, begin
        con_dh[i=1:nh], h[i] == h[i-1] + 0.5 * step * (dh[i] + dh[i-1])
        con_dv[i=1:nh], v[i] == v[i-1] + 0.5 * step * (dv[i] + dv[i-1])
        con_dm[i=1:nh], m[i] == m[i-1] + 0.5 * step * (dm[i] + dm[i-1])
    end)

    return model
end

function rocket_model_hersim(nh,coef_obj)
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
    #weight of each dimension of parameters and target residuals 
    wh = h_0
    wv = wh/step
    wm = m_0
    sum_weight = wh+wv+wm
    wh = (1-coef_obj) * wh/sum_weight
    wv = (1-coef_obj) * wv/sum_weight
    wm = (1-coef_obj) * wm/sum_weight
    @objective(model, Min, (-coef_obj)*h[nh,0] +sum( wh * (h[i+1,0] - h[i,0]-step*dh[i,0])^2 
                                                    + wv * (v[i+1,0] - v[i,0]-step*dv[i,0])^2 
                                                    + wm * (m[i+1,0] - m[i,0]-step*dm[i,0])^2 for i in 0:nh-1))    #Boundary constraints
    @constraints(model, begin
        h_ic, h[0,0] == h_0
        v_ic, v[0,0] == v_0
        m_ic, m[0,0] == m_0
        m_fc, m[nh,0] == m_f
        h_ic_1, h[0,1] == h_0
        v_ic_1, v[0,1] == v_0
        m_ic_1, m[0,1] == m_0
    end)

    #Hermite-Simpson Method
    @constraints(model,begin
        def_ref_h[i=1:nh-1], h[i,1] == 0.5 * (h[i,0] + h[i+1,0]) + 0.125 * step * (dh[i,0] - dh[i+1,0])
        def_ref_v[i=1:nh-1], v[i,1] == 0.5 * (v[i,0] + v[i+1,0]) + 0.125 * step * (dv[i,0] - dv[i+1,0])
        def_ref_m[i=1:nh-1], m[i,1] == 0.5 * (m[i,0] + m[i+1,0]) + 0.125 * step * (dm[i,0] - dm[i+1,0])

        con_dh[i=1:nh], h[i,0] == h[i-1,0] + 1/6 * step * (dh[i-1,0] + dh[i,0] + 4 * dh[i-1,1])
        con_dv[i=1:nh], v[i,0] == v[i-1,0] + 1/6 * step * (dv[i-1,0] + dv[i,0] + 4 * dv[i-1,1])
        con_dm[i=1:nh], m[i,0] == m[i-1,0] + 1/6 * step * (dm[i-1,0] + dm[i,0] + 4 * dm[i-1,1])
    end)

    return model
end

Para_list = ["nh=","tol=","coef_obj=","model:"]
model_list = [rocket_model_euler_exp,rocket_model_euler_imp,rocket_model_Traper,rocket_model_hersim]
model_list_name = ["Euler exp","Euler imp","Trapezoidal","Hermite_Simpson"]

function find_1nonzero(lst)
    return findfirst(!iszero,lst)
end

function Generate_thrust(nhs=def_nhs,tols=def_tols,objs=def_objs,models=def_models)
    conf = [nhs,tols,objs,models]
    
    #Seperate variables and parameters by identifying the type of variable
    islists_conf = [v isa AbstractVector for v in conf]
    var_index = find_1nonzero(islists_conf)
    var_name = Para_list[var_index]
    con_names = filter(x->x!=var_name,Para_list)
    con_value = [string(v) for v in conf if (v isa AbstractVector) == 0]
    if var_name != "model:"
        con_value[end] = model_list_name[parse(Int,con_value[end])]
    end
    con_title = join([join(i,"") for i in map(x->[x...],zip(con_names,con_value))],", ")
    Inputs = collect(IterTools.product(nhs,tols,objs,models))

    ##Plot by loops of variable
    P = Plots.plot(xlabel="Time", ylabel="Thrust Value",title=con_title)
    for (nh,tol,obj,num_model) in Inputs
        model = model_list[num_model](nh,obj)
        JuMP.set_optimizer(model, Ipopt.Optimizer)
        JuMP.set_attribute(model,"tol",tol)
        JuMP.set_attribute(model, "hsllib", HSL_jll.libhsl_path)
        JuMP.set_attribute(model, "linear_solver", "ma57")
        JuMP.optimize!(model)
        T_value = value.(model[:T]);
        T_Array = Array(T_value);
        if num_model ==4
            T_Array = Array(T_value[:,0]);
        end
        var_value = [nh,tol,obj,num_model][var_index]
        if var_index == 4
            var_value = model_list_name[num_model]
        end
        Plots.plot!(LinRange(0,0.2,length(T_Array)),T_Array,label="u values for $var_name $var_value")
    end

    ##Return Plots plot
    print("Loop done")
    return P
end

#Generate_thrust(250,1e-8,[0,0.1,0.5,0.9,1.0],4)
#Generate_thrust(2000,1e-8,1.0,[1,2,3,4])
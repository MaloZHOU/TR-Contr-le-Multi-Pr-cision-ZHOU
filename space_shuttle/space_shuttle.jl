# **This tutorial was originally contributed by Henrique Ferrolho.**
# Taken from: https://jump.dev/JuMP.jl/stable/tutorials/nonlinear/space_shuttle_reentry_trajectory/

using JuMP
import Interpolations

function space_shuttle_model(n)
    ## Global variables
    w = 203000.0  # weight (lb)
    g₀ = 32.174    # acceleration (ft/sec^2)
    m = w / g₀    # mass (slug)

    ## Aerodynamic and atmospheric forces on the vehicle
    ρ₀ = 0.002378
    hᵣ = 23800.0
    Rₑ = 20902900.0
    μ = 0.14076539e17
    S = 2690.0
    a₀ = -0.20704
    a₁ = 0.029244
    b₀ = 0.07854
    b₁ = -0.61592e-2
    b₂ = 0.621408e-3
    c₀ = 1.0672181
    c₁ = -0.19213774e-1
    c₂ = 0.21286289e-3
    c₃ = -0.10117249e-5

    ## Initial conditions
    h_s = 2.6          # altitude (ft) / 1e5
    ϕ_s = deg2rad(0)   # longitude (rad)
    θ_s = deg2rad(0)   # latitude (rad)
    v_s = 2.56         # velocity (ft/sec) / 1e4
    γ_s = deg2rad(-1)  # flight path angle (rad)
    ψ_s = deg2rad(90)  # azimuth (rad)
    α_s = deg2rad(0)   # angle of attack (rad)
    β_s = deg2rad(0)   # bank angle (rad)
    t_s = 1.00         # time step (sec)

    ## Final conditions, the so-called Terminal Area Energy Management (TAEM)
    h_t = 0.8          # altitude (ft) / 1e5
    v_t = 0.25         # velocity (ft/sec) / 1e4
    γ_t = deg2rad(-5)  # flight path angle (rad)

    model = Model()

    @variables(model, begin
        0 ≤ scaled_h[1:n]                # altitude (ft) / 1e5
        ϕ[1:n]                # longitude (rad)
        deg2rad(-89) ≤ θ[1:n] ≤ deg2rad(89)  # latitude (rad)
        1e-4 ≤ scaled_v[1:n]                # velocity (ft/sec) / 1e4
        deg2rad(-89) ≤ γ[1:n] ≤ deg2rad(89)  # flight path angle (rad)
        ψ[1:n]                # azimuth (rad)
        deg2rad(-90) ≤ α[1:n] ≤ deg2rad(90)  # angle of attack (rad)
        deg2rad(-89) ≤ β[1:n] ≤ deg2rad(1)  # bank angle (rad)
        ##        3.5 ≤       Δt[1:n] ≤ 4.5          # time step (sec)
        Δt[1:n] == 4.0         # time step (sec)
    end);

    ## Fix initial conditions
    @constraints(model, begin
        scaled_h[1] == h_s
        ϕ[1] == ϕ_s
        θ[1] == θ_s
        scaled_v[1] == v_s
        γ[1] == γ_s
        ψ[1] == ψ_s
        scaled_h[n] == h_t
        scaled_v[n] == v_t
        γ[n] == γ_t
    end)


    ## Initial guess: linear interpolation between boundary conditions
    x_s = [h_s, ϕ_s, θ_s, v_s, γ_s, ψ_s, α_s, β_s, t_s]
    x_t = [h_t, ϕ_s, θ_s, v_t, γ_t, ψ_s, α_s, β_s, t_s]
    interp_linear = Interpolations.LinearInterpolation([1, n], [x_s, x_t])
    initial_guess = mapreduce(transpose, vcat, interp_linear.(1:n))
    set_start_value.(all_variables(model), vec(initial_guess))

    ## Functions to restore `h` and `v` to their true scale
    @expression(model, h[j = 1:n], scaled_h[j] * 1e5)
    @expression(model, v[j = 1:n], scaled_v[j] * 1e4)

    ## Helper functions
    @expression(model, c_L[j = 1:n], a₀ + a₁ * rad2deg(α[j]))
    @expression(model, c_D[j = 1:n], b₀ + b₁ * rad2deg(α[j]) + b₂ * rad2deg(α[j])^2)
    @expression(model, ρ[j = 1:n], ρ₀ * exp(-h[j] / hᵣ))
    @expression(model, D[j = 1:n], 0.5 * c_D[j] * S * ρ[j] * v[j]^2)
    @expression(model, L[j = 1:n], 0.5 * c_L[j] * S * ρ[j] * v[j]^2)
    @expression(model, r[j = 1:n], Rₑ + h[j])
    @expression(model, g[j = 1:n], μ / r[j]^2)

    ## Motion of the vehicle as a differential-algebraic system of equations (DAEs)
    @expression(model, δh[j = 1:n], v[j] * sin(γ[j]))
    @expression(
        model,
        δϕ[j = 1:n],
        (v[j] / r[j]) * cos(γ[j]) * sin(ψ[j]) / cos(θ[j])
    )
    @expression(model, δθ[j = 1:n], (v[j] / r[j]) * cos(γ[j]) * cos(ψ[j]))
    @expression(model, δv[j = 1:n], -(D[j] / m) - g[j] * sin(γ[j]))
    @expression(
        model,
        δγ[j = 1:n],
        (L[j] / (m * v[j])) * cos(β[j]) +
        cos(γ[j]) * ((v[j] / r[j]) - (g[j] / v[j]))
    )
    @expression(
        model,
        δψ[j = 1:n],
        (1 / (m * v[j] * cos(γ[j]))) * L[j] * sin(β[j]) +
        (v[j] / (r[j] * cos(θ[j]))) * cos(γ[j]) * sin(ψ[j]) * sin(θ[j])
    )

    ## System dynamics, trapezoidal method
    @constraints(model, begin
        [j=2:n], h[j] == h[j-1] + 0.5 * Δt[j-1] * (δh[j] + δh[j-1])
        [j=2:n], ϕ[j] == ϕ[j-1] + 0.5 * Δt[j-1] * (δϕ[j] + δϕ[j-1])
        [j=2:n], θ[j] == θ[j-1] + 0.5 * Δt[j-1] * (δθ[j] + δθ[j-1])
        [j=2:n], v[j] == v[j-1] + 0.5 * Δt[j-1] * (δv[j] + δv[j-1])
        [j=2:n], γ[j] == γ[j-1] + 0.5 * Δt[j-1] * (δγ[j] + δγ[j-1])
        [j=2:n], ψ[j] == ψ[j-1] + 0.5 * Δt[j-1] * (δψ[j] + δψ[j-1])
    end)

    ## Objective: Maximize cross-range
    @objective(model, Min, -θ[n])
    return model
end


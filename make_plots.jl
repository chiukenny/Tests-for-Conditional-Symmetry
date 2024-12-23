## USAGE
## -----
##
## From the command line, execute the following command to
## make the plots and figures in the paper after running all experiments:
##     julia make_plots.jl
##
## Edit the 'Paths' variables below as necessary


# Paths
dir_src = "./src/"               # Source code directory
dir_out = "./outputs/"           # Output directory
dir_out_dat = "./outputs/data/"  # Cleaned data directory
dir_plt = "./outputs/plots/"     # Output plot directory


# Packages and modules
# --------------------

println("Loading packages and modules")

using Random
using Distributions
using DataFrames
using CSV
using Plots
using LaTeXStrings
using Measures
using Colors
using ColorSchemes
using HDF5
import H5Zblosc

include(dir_src * "global_variables.jl")  # Global variables
include(dir_src * "util.jl")              # Shared functions
include(dir_src * "groups.jl")            # Groups and related functions


# Default plot settings and convenience plots
# -------------------------------------------

default(fontfamily="Computer Modern", framestyle=:box, label=nothing, grid=false, legend=:none,
        linewidth=3, titlefontsize=8, guidefontsize=8, tickfontsize=6, legendfontsize=6,
        margin=0mm, dpi=300)

# Set default palette to be CVD-friendly
iGV_BS_FS = "indep_" * GV_BS_FS
iGV_BS_SK = "indep_" * GV_BS_SK
iGV_RN_FS = "indep_" * GV_RN_FS
iGV_RN_SK = "indep_" * GV_RN_SK
rGV_BS_FS = "reuse_" * GV_BS_FS
rGV_BS_SK = "reuse_" * GV_BS_SK
rGV_RN_FS = "reuse_" * GV_RN_FS
rGV_RN_SK = "reuse_" * GV_RN_SK
iGV_FUSE = "indep_" * GV_FUSE
iGV_SK = "indep_" * GV_SK
rGV_FUSE = "reuse_" * GV_FUSE
rGV_SK = "reuse_" * GV_SK
gr(palette = :tol_bright)
colsDict = Dict(iGV_BS_FS=>1, iGV_BS_SK=>"darkorange2", iGV_RN_FS=>5,  iGV_RN_SK=>"goldenrod1",
                rGV_BS_FS=>1, rGV_BS_SK=>"darkorange2", rGV_RN_FS=>5,  rGV_RN_SK=>"goldenrod1",
                GV_BS_FS=>1, GV_BS_SK=>"darkorange2", GV_RN_FS=>5,  GV_RN_SK=>"goldenrod1", GV_FUSE=>5, GV_SK=>"goldenrod1")

# Axis label hack
function shared_xlab(label; top_margin=2mm, bottom_margin=0mm)
    return scatter([0], [0], xlims=[0,1], xmirror=true, xticks=[], yticks=[], alpha=0, xlab=label, border=:grid,
                   top_margin=top_margin, bottom_margin=bottom_margin)
end
function shared_ylab(label; right_margin=3mm)
    return scatter([0], [0], ylims=[0,1], ymirror=true, xticks=[], yticks=[], alpha=0, ylab=label, border=:grid, right_margin=right_margin)
end

# Blank plot
ep = plot(framestyle=:none, ticks=[])

# Tests
all_tests = [iGV_BS_FS, iGV_BS_SK, rGV_BS_FS, rGV_BS_SK, iGV_RN_FS, iGV_RN_SK,rGV_RN_FS, rGV_RN_SK]
all_tests_i = [iGV_BS_FS, iGV_BS_SK, iGV_RN_FS, iGV_RN_SK]
all_tests_r = [rGV_BS_FS, rGV_BS_SK, rGV_RN_FS, rGV_RN_SK]
test_labs = [GV_BS_FS, GV_BS_SK, GV_RN_FS, GV_RN_SK]

# Default legend
n_labs = 4
leg = plot(zeros(1,n_labs), showaxis=false, legend=true, label=reshape(test_labs,1,n_labs), legendcolumns=Int(n_labs/2),
           foreground_color_legend=nothing, color=reshape([colsDict[test] for test in all_tests_r],1,n_labs))


# Plotting functions
# ------------------

# Computes rejection rate and standard deviations
function compute_rej_rate(df::DataFrame, tests::AbstractVector{String}, means::AbstractVector{Float64}, sds::AbstractVector{Float64})
    n = size(df, 1)
    @views @inbounds for i in 1:length(tests)
        test = tests[i]
        means[i] = mean.(eachcol(df[:,Regex(test*"_p\$")] .< α))[1]
        sds[i] = sqrt.(means[i] .* (1 .- means[i]) / n)
    end
end

# Plots rejection rate and standard deviations
function plot_rej_rate(means::AbstractMatrix{Float64}, sds::AbstractMatrix{Float64},
                       tests::AbstractVector{String}, x::AbstractVector{<:Number};
                       title="", xlab="", ylab="", ylims=(0,1), pl_xticks=false, pl_yticks=false, all_tests=all_tests, xrotation=0)
    p = plot(xlab=xlab, ylab=ylab, xlims=(minimum(x),maximum(x)), ylims=ylims, grid=true, title=title, xrotation=xrotation)
    @views @inbounds for test in tests
        i = findall(x->x==test, all_tests)[1]
        plot!(x, means[i,:], linecolor=colsDict[test], linealpha=0.7, label=test)
        plot!(x, max.(0,means[i,:].-sds[i,:]), fillrange=min.(1,means[i,:].+sds[i,:]),
              label=:none, fillalpha=0.2, linealpha=0, fillcolor=colsDict[test])
    end
    if !pl_xticks
        plot!(xticks = (xticks(p)[1][1],""))
    end
    if !pl_yticks
        plot!(yticks = (yticks(p)[1][1],""))
    end
    return p
end

# Plots p-value distribution
function hist(df::DataFrame, test::String, ylab=""; bins=10, bw=0.1, x_ticks=[0,0.5,1], title="", last_row=false)
    x = df[:, Regex(test*"_p\$")][!,1]
    xx = vcat(x, collect(minimum(x_ticks):maximum(x_ticks))./bins)  # Visual hack to force all bins to show
    xt = last_row ? [x_ticks[i]==0 ? "0" : rstrip(string(x_ticks[i]),['0','.']) for i in 1:length(x_ticks)] : []
    p = histogram(xx, bins=bins, fillcolor=colsDict[test], linecolor=colsDict[test], lw=1, bar_width=bw,
                  xticks=(x_ticks,xt), xlims=[minimum(x_ticks),maximum(x_ticks)], ylab=ylab, yticks=[], title=title)
    yl = ylims(p)
    ylims!(0, yl[2])
    return p
end

# Creates a circle
function circle(h, k, r)
    θ = LinRange(0, 2*π, 200)
    h .+ r*sin.(θ), k .+ r*cos.(θ)
end


## Plots

# Symmetry figure
# ---------------

Random.seed!(1)

G = Group(f_sample=rand_θ, f_transform=rotate_2D)
n = 250
Pz = Normal(1, 0.2)

# Invariance vs non-invariance
x = zeros(2, n)
x[1,:] = rand(Pz, n)
x = transform_all(G, x)
p1 = plot(0, 0)
hline!([0], color=:black, linealpha=0.1, linewidth=1)
vline!([0], color=:black, linealpha=0.1, linewidth=1)
scatter!(x[1,:], x[2,:], markercolor=:hotpink, title="Marginally invariant", markersize=3, markerstrokewidth=0, markeralpha=0.35)

x = zeros(2, n)
x[1,:] = rand(Pz, n)
g = Vector{Float64}(undef, n)
for i in 1:n
    g[i] = rand_θ()
    while abs(g[i]) > 5*π/3
        g[i] = rand_θ()
    end
end
x = transform_each(G, x, g)
p2 = plot(0, 0)
hline!([0], color=:black, linealpha=0.1, linewidth=1)
vline!([0], color=:black, linealpha=0.1, linewidth=1)
scatter!(x[1,:], x[2,:], markercolor=:hotpink, title="Marginally non-invariant", markersize=3, markerstrokewidth=0, markeralpha=0.35)

# Equivariance vs non-equivariance
n = 100
x = [0 0 -1; 1. -1. 0]
p_x = copy(x)
p_x[:,2] = rotate_2D(x[:,2], π/4)
col1 = :darkorchid
col2 = :orange
col3 = :green3

# Equivariance
Random.seed!(1)
Σ = [0.2 0; 0 0.04]
P = MvNormal(zeros(2), Σ)
Σ3 = [0.04 0; 0 0.2]
P3 = MvNormal(zeros(2), Σ3)
# Y | X1 and Y | X2
y1 = x[:,1] + [0; 1.25] .+ rand(P, n)
y2 = transform_all(G, x[:,2]-[0;1.25].+rand(P,n), π/4)
y3 = x[:,3] - [1.25;0] .+ rand(P3,n)

p3 = plot(circle(0,0,1), seriestype=[:shape], lw=1, fillalpha=0, linealpha=0.04, linecolor=:black, title="Conditionally equivariant")
plot!(circle(0,0,2.25), seriestype=[:shape], lw=1, fillalpha=0, linealpha=0.04, linecolor=:black)
hline!([0], color=:black, linealpha=0.1, linewidth=1)
vline!([0], color=:black, linealpha=0.1, linewidth=1)
scatter!(y1[1,:], y1[2,:], markeralpha=0.35, color=col1, markerstrokewidth=0, markersize=3, markershape=:utriangle)
scatter!(y2[1,:], y2[2,:], markeralpha=0.35, color=col2, markerstrokewidth=0, markersize=3, markershape=:utriangle)
scatter!(y3[1,:], y3[2,:], markeralpha=0.35, color=col3, markerstrokewidth=0, markersize=3, markershape=:utriangle)
scatter!(p_x[1,:], p_x[2,:], markeralpha=0.5, color=[col1,col2,col3], markersize=5)

# Not equivariance
Random.seed!(1)
# Y | X1
Σ1 = [0.2 0; 0 0.04]
P1 = MvNormal(zeros(2), Σ1)
y1 = x[:,1] + [0;1.25] .+ rand(P1,n)
# Y | X2
Σ2 = [0.04 0; 0 0.2]
P2 = MvNormal(zeros(2), Σ2)
y2 = transform_all(G, x[:,2]-[0;1.25].+rand(P2,n), π/4)
# Y | X3
Σ3 = [0.2 0; 0 0.2]
P3 = MvNormal(zeros(2), Σ3)
y3 = x[:,3] - [1.25;0] .+ rand(P3,n)

p4 = plot(circle(0,0,1), seriestype=[:shape], lw=1, fillalpha=0, linealpha=0.04, linecolor=:black, title="Conditionally non-equivariant")
plot!(circle(0,0,2.25), seriestype=[:shape], lw=1, fillalpha=0, linealpha=0.04, linecolor=:black)
hline!([0], color=:black, linealpha=0.1, linewidth=1)
vline!([0], color=:black, linealpha=0.1, linewidth=1)
scatter!(y1[1,:], y1[2,:], markeralpha=0.35, color=col1, markerstrokewidth=0, markersize=3, markershape=:utriangle)
scatter!(y2[1,:], y2[2,:], markeralpha=0.35, color=col2, markerstrokewidth=0, markersize=3, markershape=:utriangle)
scatter!(y3[1,:], y3[2,:], markeralpha=0.35, color=col3, markerstrokewidth=0, markersize=3, markershape=:utriangle)
scatter!(p_x[1,:], p_x[2,:], markeralpha=0.5, color=[col1,col2,col3], markersize=5)

# Conditional invariance vs non-conditional invariance
Random.seed!(1)
Σ = [0.04 0; 0 0.2]
P = MvNormal(zeros(2), Σ)
# Y | X1, Y | X2, Y | X3
y1 = [2.25;0] .+ rand(P, n)
y2 = [2.25;0] .+ rand(P, n)
y3 = [2.25;0] .+ rand(P, n)

p5 = plot(circle(0,0,1), seriestype=[:shape], lw=1, fillalpha=0, linealpha=0.04, linecolor=:black, title="Conditionally invariant")
plot!(circle(0,0,2.25), seriestype=[:shape], lw=1, fillalpha=0, linealpha=0.04, linecolor=:black)
hline!([0], color=:black, linealpha=0.1, linewidth=1)
vline!([0], color=:black, linealpha=0.1, linewidth=1)
scatter!(y3[1,:], y3[2,:], markeralpha=0.35, color=col3, markerstrokewidth=0, markersize=3, markershape=:utriangle)
scatter!(y1[1,:], y1[2,:], markeralpha=0.35, color=col1, markerstrokewidth=0, markersize=3, markershape=:utriangle)
scatter!(y2[1,:], y2[2,:], markeralpha=0.35, color=col2, markerstrokewidth=0, markersize=3, markershape=:utriangle)
scatter!(p_x[1,:], p_x[2,:], markeralpha=0.5, color=[col1,col2,col3], markersize=5)

Random.seed!(1)
Σ = [0.2 0; 0 0.04]
P = MvNormal(zeros(2), Σ)
Σ3 = [0.04 0; 0 0.2]
P3 = MvNormal(zeros(2), Σ3)
# Y | X1 and Y | X2
y1 = x[:,1] + [0; 1.25] .+ rand(P, n)
y2 = transform_all(G, x[:,2]-[0;1.25].+rand(P,n), π/4)
y3 = x[:,3] - [1.25;0] .+ rand(P3,n)

p6 = plot(circle(0,0,1), seriestype=[:shape], lw=1, fillalpha=0, linealpha=0.04, linecolor=:black, title="Conditionally non-invariant")
plot!(circle(0,0,2.25), seriestype=[:shape], lw=1, fillalpha=0, linealpha=0.04, linecolor=:black)
hline!([0], color=:black, linealpha=0.1, linewidth=1)
vline!([0], color=:black, linealpha=0.1, linewidth=1)
scatter!(y1[1,:], y1[2,:], markeralpha=0.35, color=col1, markerstrokewidth=0, markersize=3, markershape=:utriangle)
scatter!(y2[1,:], y2[2,:], markeralpha=0.35, color=col2, markerstrokewidth=0, markersize=3, markershape=:utriangle)
scatter!(y3[1,:], y3[2,:], markeralpha=0.35, color=col3, markerstrokewidth=0, markersize=3, markershape=:utriangle)
scatter!(p_x[1,:], p_x[2,:], markeralpha=0.5, color=[col1,col2,col3], markersize=5)

fig1 = plot(p1, p3, p5, p2, p4, p6, layout=grid(2,3), size=(525,375), legend=false, axis=false, aspect_ratio=1,
           xlim=(-3,3), ylim=(-3,3), colorbar=false, bottom_margin=-3mm)
fig1_name = dir_plt * "symmetries.pdf"
savefig(fig1, fig1_name)
println("Created $(fig1_name)")


# Equivariance figure
# -------------------

n = 100
x = [0 0 -1.; 1. -1. 0]
p_x = copy(x)
p_x[:,2] = rotate_2D(x[:,2], π/4)
ρx = [1 1 1; 0 0 0]
col1 = :darkorchid
col2 = :orange
col3 = :green3

# Equivariant
Random.seed!(1)
Σ = [0.2 0; 0 0.04]
P = MvNormal(zeros(2), Σ)
Σ3 = [0.04 0; 0 0.2]
P3 = MvNormal(zeros(2), Σ3)
# Y | X1 and Y | X2
y1 = x[:,1] + [0; 1.25] .+ rand(P, n)
y2 = transform_all(G, x[:,2]-[0;1.25].+rand(P,n), π/4)
y3 = x[:,3] - [1.25;0] .+ rand(P3,n)

p1 = plot(circle(0,0,1), seriestype=[:shape], lw=1, fillalpha=0, linealpha=0.04, linecolor=:black,
          title=L"P_{Y|X}(\:\bullet\mid x)", ylab="Equivariant")
plot!(circle(0,0,2.25), seriestype=[:shape], lw=1, fillalpha=0, linealpha=0.04, linecolor=:black)
hline!([0], color=:black, linealpha=0.1, linewidth=1)
vline!([0], color=:black, linealpha=0.1, linewidth=1)
scatter!(y3[1,:], y3[2,:], markeralpha=0.35, color=col3, markerstrokewidth=0, markersize=3, markershape=:utriangle)
scatter!(y1[1,:], y1[2,:], markeralpha=0.35, color=col1, markerstrokewidth=0, markersize=3, markershape=:utriangle)
scatter!(y2[1,:], y2[2,:], markeralpha=0.35, color=col2, markerstrokewidth=0, markersize=3, markershape=:utriangle)
scatter!(p_x[1,:], p_x[2,:], markeralpha=0.5, color=[col1,col2,col3], markersize=5)

τy1 = similar(y1)
τy2 = similar(y2)
τy3 = similar(y2)
for i in 1:n
    τy1[:,i] = rotate_2D(y1[:,i], 3*π/2)
    τy2[:,i] = rotate_2D(y2[:,i], π/4)
    τy3[:,i] = rotate_2D(y3[:,i], float(π))
end
p2 = plot(circle(0,0,1), seriestype=[:shape], lw=1, fillalpha=0, linealpha=0.04, linecolor=:black,
          title=L"\tau(x)_*P_{Y|X}(\:\bullet\mid x)")
plot!(circle(0,0,2.25), seriestype=[:shape], lw=1, fillalpha=0, linealpha=0.04, linecolor=:black)
hline!([0], color=:black, linealpha=0.1, linewidth=1)
vline!([0], color=:black, linealpha=0.1, linewidth=1)
scatter!(τy3[1,:], τy3[2,:], markeralpha=0.35, color=col3, markerstrokewidth=0, markersize=3, markershape=:utriangle)
scatter!(τy1[1,:], τy1[2,:], markeralpha=0.35, color=col1, markerstrokewidth=0, markersize=3, markershape=:utriangle)
scatter!(τy2[1,:], τy2[2,:], markeralpha=0.35, color=col2, markerstrokewidth=0, markersize=3, markershape=:utriangle)
scatter!(ρx[1,:], ρx[2,:], markeralpha=0.5, color=[col1,col2,col3], markersize=5)

# Non-equivariance
Random.seed!(1)
# Y | X1
Σ1 = [0.2 0; 0 0.04]
P1 = MvNormal(zeros(2), Σ1)
y1 = x[:,1] + [0;1.25] .+ rand(P1,n)
# Y | X2
Σ2 = [0.04 0; 0 0.2]
P2 = MvNormal(zeros(2), Σ2)
y2 = transform_all(G, x[:,2]-[0;1.25].+rand(P2,n), π/4)
# Y | X3
Σ3 = [0.2 0; 0 0.2]
P3 = MvNormal(zeros(2), Σ3)
y3 = x[:,3] - [1.25;0] .+ rand(P3,n)

p3 = plot(circle(0,0,1), seriestype=[:shape], lw=1, fillalpha=0, linealpha=0.04, linecolor=:black, ylab="Non-equivariant")
plot!(circle(0,0,2.25), seriestype=[:shape], lw=1, fillalpha=0, linealpha=0.04, linecolor=:black)
hline!([0], color=:black, linealpha=0.1, linewidth=1)
vline!([0], color=:black, linealpha=0.1, linewidth=1)
scatter!(y3[1,:], y3[2,:], markeralpha=0.35, color=col3, markerstrokewidth=0, markersize=3, markershape=:utriangle)
scatter!(y1[1,:], y1[2,:], markeralpha=0.35, color=col1, markerstrokewidth=0, markersize=3, markershape=:utriangle)
scatter!(y2[1,:], y2[2,:], markeralpha=0.35, color=col2, markerstrokewidth=0, markersize=3, markershape=:utriangle)
scatter!(p_x[1,:], p_x[2,:], markeralpha=0.5, color=[col1,col2,col3], markersize=5)

for i in 1:n
    τy1[:,i] = rotate_2D(y1[:,i], 3*π/2)
    τy2[:,i] = rotate_2D(y2[:,i], π/4)
    τy3[:,i] = rotate_2D(y3[:,i], float(π))
end
p4 = plot(circle(0,0,1), seriestype=[:shape], lw=1, fillalpha=0, linealpha=0.04, linecolor=:black)
plot!(circle(0,0,2.25), seriestype=[:shape], lw=1, fillalpha=0, linealpha=0.04, linecolor=:black)
hline!([0], color=:black, linealpha=0.1, linewidth=1)
vline!([0], color=:black, linealpha=0.1, linewidth=1)
scatter!(τy3[1,:], τy3[2,:], markeralpha=0.35, color=col3, markerstrokewidth=0, markersize=3, markershape=:utriangle)
scatter!(τy1[1,:], τy1[2,:], markeralpha=0.35, color=col1, markerstrokewidth=0, markersize=3, markershape=:utriangle)
scatter!(τy2[1,:], τy2[2,:], markeralpha=0.35, color=col2, markerstrokewidth=0, markersize=3, markershape=:utriangle)
scatter!(ρx[1,:], ρx[2,:], markeralpha=0.5, color=[col1,col2,col3], markersize=5)

fig2 = plot(p1, p2, p3, p4, layout=grid(2,2), size=(350,300), bottom_margin=-5mm,
           xaxis=false, yaxis=false, aspect_ratio=1, xlims=(-3,3), ylims=(-3,3))
fig2_name = dir_plt * "equivariance.pdf"
savefig(fig2, fig2_name)
println("Created $(fig2_name)")


# Synthetic experiment: changing covariance
# -----------------------------------------

println("Creating changing covariance experiment plots")

# Set up experiment parameters
ps = GV_GAUSS_COV_p
n_p = length(ps)

ns = GV_GAUSS_COV_n
n_n = length(ns)

ds = GV_GAUSS_COV_d
n_d = length(ds)

# Read experiment results
n_tests = length(all_tests)
means = Array{Float64}(undef, n_tests, n_n, n_p, n_d)
sds = Array{Float64}(undef, n_tests, n_n, n_p, n_d)
@views @inbounds @threads for i in 1:n_d
    for j in 1:n_p
        for k in 1:n_n
            pvals = CSV.read(dir_out*"gauss_cov_SO$(ds[i])_N1000_B100_n$(ns[k])_p$(ps[j]).csv", DataFrame)
            compute_rej_rate(pvals, all_tests, means[:,k,j,i], sds[:,k,j,i])
        end
    end
end

# Make plots
figs = Matrix{Any}(undef, n_p, n_d)
@views @inbounds for i in 1:n_d
    for j in 1:n_p
        xticks = i==n_d
        yticks = j==1
        ylab = j==1 ? L"d=%$(ds[i])" : ""
        title = i==1 ? L"p=%$(min(ps[j],0.99))" : ""
        figs[j,i] = plot_rej_rate(means[:,:,j,i], sds[:,:,j,i], all_tests_r, ns,
                                  ylab=ylab, title=title, pl_xticks=xticks, pl_yticks=yticks)
    end
end
p_xlab = shared_xlab(L"n", top_margin=0mm)
p_ylab = shared_ylab("Reject rate", right_margin=4.5mm)
l = @layout [c{0.0001w} [a{0.03h} ; grid(n_d,n_p) ; b{0.03h}]]
fig = plot(p_ylab, leg,
                   figs...,
                   p_xlab, layout=l, size=(600,325))
fig_name = dir_plt * "gauss_equiv_cov.pdf"
savefig(fig, fig_name)
println("Created $(fig_name)")

@views @inbounds for i in 1:n_d
    for j in 1:n_p
        xticks = i==n_d
        yticks = j==1
        ylab = j==1 ? L"d=%$(ds[i])" : ""
        title = i==1 ? L"p=%$(min(ps[j],0.99))" : ""
        figs[j,i] = plot_rej_rate_all(means[:,:,j,i], sds[:,:,j,i], all_tests, ns,
                                      ylab=ylab, title=title, pl_xticks=xticks, pl_yticks=yticks)
    end
end
p_xlab = shared_xlab(L"n", top_margin=0mm)
p_ylab = shared_ylab("Reject rate", right_margin=4.5mm)
l = @layout [c{0.0001w} [a{0.03h} ; grid(n_d,n_p) ; b{0.03h}]]
fig = plot(p_ylab, leg,
                   figs...,
                   p_xlab, layout=l, size=(600,325))
fig_name = dir_plt * "gauss_equiv_cov_all.pdf"
savefig(fig, fig_name)
println("Created $(fig_name)")


# Synthetic experiment: isolated non-equivariance
# -----------------------------------------------

println("Creating isolated non-equivariance experiment plots")

# Set up experiment parameters
ps = GV_GAUSS_COV_p[2:end]
n_p = length(ps)

ns = GV_GAUSS_COV_n
n_n = length(ns)

ds = GV_GAUSS_COV_d
n_d = length(ds)

# Read experiment results
n_tests = length(all_tests)
means = Array{Float64}(undef, n_tests, n_n, n_p, n_d)
sds = Array{Float64}(undef, n_tests, n_n, n_p, n_d)
@views @inbounds @threads for i in 1:n_d
    for j in 1:n_p
        for k in 1:n_n
            pvals = CSV.read(dir_out*"gauss_nonequiv_cov_SO$(ds[i])_N1000_B100_n$(ns[k])_p$(ps[j]).csv", DataFrame)
            compute_rej_rate(pvals, all_tests, means[:,k,j,i], sds[:,k,j,i])
        end
    end
end

# Make plots
figs = Matrix{Any}(undef, n_p, n_d)
@views @inbounds for i in 1:n_d
    for j in 1:n_p
        xticks = i==n_d
        yticks = j==1
        ylab = j==1 ? L"d=%$(ds[i])" : ""
        title = i==1 ? L"p=%$(min(ps[j],0.99))" : ""
        figs[j,i] = plot_rej_rate(means[:,:,j,i], sds[:,:,j,i], all_tests_r, ns,
                                  ylab=ylab, title=title, pl_xticks=xticks, pl_yticks=yticks)
    end
end
p_xlab = shared_xlab(L"n", top_margin=0mm)
p_ylab = shared_ylab("Reject rate", right_margin=4.5mm)
l = @layout [c{0.0001w} [a{0.03h} ; grid(n_d,n_p) ; b{0.03h}]]
fig = plot(p_ylab, leg,
                   figs...,
                   p_xlab, layout=l, size=(500,325))
fig_name = dir_plt * "gauss_nonequiv_cov.pdf"
savefig(fig, fig_name)
println("Created $(fig_name)")

@views @inbounds for i in 1:n_d
    for j in 1:n_p
        xticks = i==n_d
        yticks = j==1
        ylab = j==1 ? L"d=%$(ds[i])" : ""
        title = i==1 ? L"p=%$(min(ps[j],0.99))" : ""
        figs[j,i] = plot_rej_rate_all(means[:,:,j,i], sds[:,:,j,i], all_tests, ns,
                                      ylab=ylab, title=title, pl_xticks=xticks, pl_yticks=yticks)
    end
end
p_xlab = shared_xlab(L"n", top_margin=0mm)
p_ylab = shared_ylab("Reject rate", right_margin=4.5mm)
l = @layout [c{0.0001w} [a{0.03h} ; grid(n_d,n_p) ; b{0.03h}]]
fig = plot(p_ylab, leg,
                   figs...,
                   p_xlab, layout=l, size=(500,325))
fig_name = dir_plt * "gauss_nonequiv_cov_all.pdf"
savefig(fig, fig_name)
println("Created $(fig_name)")


# Synthetic experiment: approximate versus exact conditional sampling
# -------------------------------------------------------------------

println("Creating approximate vs exact conditional sampling experiment plots")

# Set up experiment parameters
ps = GV_GAUSS_TRUTH_p
n_p = length(ps)

ns = GV_GAUSS_TRUTH_n
n_n = length(ns)

ds = GV_GAUSS_TRUTH_d
n_d = length(ds)

es = ["", "_truth"]
n_e = length(es)

# Read experiment results
n_tests = length(all_tests)
means = Array{Float64}(undef, n_tests, n_n, n_p, n_d, n_e)
sds = Array{Float64}(undef, n_tests, n_n, n_p, n_d, n_e)
@views @inbounds @threads for e in 1:n_e
    for i in 1:n_d
        for j in 1:n_p
            for k in 1:n_n
                pvals = CSV.read(dir_out*"gauss_truth$(es[e])_N1000_B100_n$(ns[k])_p$(ps[j])_d$(ds[i]).csv", DataFrame)
                compute_rej_rate(pvals, all_tests, means[:,k,j,i,e], sds[:,k,j,i,e])
            end
        end
    end
end

# Make plots
function plot_truth(means::AbstractArray{Float64}, sds::AbstractArray{Float64},
                    tests::AbstractVector{String}, x::AbstractVector{<:Number};
                    title="", xlab="", ylab="", ylims=(0,1), pl_xticks=false, pl_yticks=false, all_tests=all_tests)
    p = plot(xlab=xlab, ylab=ylab, xlims=(minimum(x),maximum(x)), ylims=ylims, grid=true, title=title)
    @views @inbounds for test in tests
        i = findall(x->x==test, all_tests)[1]
        plot!(x, means[i,:,2], linecolor=colsDict[test], linealpha=1, linewidth=2, linestyle=:dot, label=test)
        plot!(x, means[i,:,1], linecolor=colsDict[test], linealpha=0.5, linewidth=2, label=test)
    end
    if !pl_xticks
        plot!(xticks = (xticks(p)[1][1],""))
    end
    if !pl_yticks
        plot!(yticks = (yticks(p)[1][1],""))
    end
    return p
end
figsr = Matrix{Any}(undef, n_p, n_d)
figsi = Matrix{Any}(undef, n_p, n_d)
@views @inbounds for i in 1:n_d
    for j in 1:n_p
        xticks = i==n_d
        yticks = j==1
        ylab = j==1 ? L"d=%$(ds[i])" : ""
        title = i==1 ? L"p=%$(min(ps[j],0.99))" : ""
        figsr[j,i] = plot_truth(means[:,:,j,i,:], sds[:,:,j,i,:], all_tests_r, ns,
                                ylab=ylab, title=title, pl_xticks=xticks, pl_yticks=yticks)
        figsi[j,i] = plot_truth(means[:,:,j,i,:], sds[:,:,j,i,:], all_tests_i, ns,
                                ylab=ylab, title=title, pl_xticks=xticks, pl_yticks=yticks)
    end
end
p_xlab = shared_xlab(L"n", top_margin=0mm)
p_ylab = shared_ylab("Reject rate", right_margin=4.5mm)
l = @layout [c{0.0001w} [a{0.03h} ; grid(n_d,n_p) ; b{0.03h}]]
figr = plot(p_ylab, leg,
                    figsr...,
                    p_xlab, layout=l, size=(600,325))
figr_name = dir_plt * "gauss_equiv_truth_reuse.pdf"
savefig(figr, figr_name)
println("Created $(figr_name)")

p_xlab = shared_xlab(L"n", top_margin=0mm)
p_ylab = shared_ylab("Reject rate", right_margin=4.5mm)
l = @layout [c{0.0001w} [a{0.03h} ; grid(n_d,n_p) ; b{0.03h}]]
figi = plot(p_ylab, leg,
                    figsi...,
                    p_xlab, layout=l, size=(600,325))
figi_name = dir_plt * "gauss_equiv_truth_indep.pdf"
savefig(figi, figi_name)
println("Created $(figi_name)")

# p-value plots
n_tests = length(test_labs)
figsi = Matrix{Any}(undef, n_p, n_tests)
figsr = Matrix{Any}(undef, n_p, n_tests)
figsi_t = Matrix{Any}(undef, n_p, n_tests)
figsr_t = Matrix{Any}(undef, n_p, n_tests)
@views @inbounds @threads for i in 1:n_p
    pvals = CSV.read(dir_out*"gauss_truth_N1000_B100_n250_p$(ps[i])_d3.csv", DataFrame)
    for j in 1:n_tests
        last_row = j==n_tests
        title = j==1 ? L"p=%$(min(ps[i],0.99))" : ""
        ylab = i==1 ? test_labs[j] : ""
        figsi[i,j] = hist(pvals, all_tests_i[j], ylab, title=title, last_row=last_row)
        figsr[i,j] = hist(pvals, all_tests_r[j], ylab, title=title, last_row=last_row)
    end
    pvals = CSV.read(dir_out*"gauss_truth_truth_N1000_B100_n250_p$(ps[i])_d3.csv", DataFrame)
    for j in 1:n_tests
        last_row = j==n_tests
        title = j==1 ? L"p=%$(min(ps[i],0.99))" : ""
        ylab = i==1 ? test_labs[j] : ""
        figsi_t[i,j] = hist(pvals, all_tests_i[j], ylab, title=title, last_row=last_row)
        figsr_t[i,j] = hist(pvals, all_tests_r[j], ylab, title=title, last_row=last_row)
    end
end

p_title = shared_xlab("Independent comparison sets (Approx. sampling)", bottom_margin=-3mm)
p_xlab = shared_xlab(L"$p$-value", top_margin=-1mm)
p_ylab = shared_ylab("Frequency", right_margin=6mm)
l = @layout [c{0.0001w} [c{0.03h} ; grid(n_tests,n_p) ; b{0.03h}]]
figi = plot(p_ylab, p_title, figsi..., p_xlab, layout=l, size=(450,300))
figi_name = dir_plt * "pval_gauss_equiv_truth_indep_apx.pdf"
savefig(figi, figi_name)

p_title = shared_xlab("Independent comparison sets (Exact sampling)", bottom_margin=-3mm)
p_xlab = shared_xlab(L"$p$-value", top_margin=-1mm)
p_ylab = shared_ylab("Frequency", right_margin=6mm)
l = @layout [c{0.0001w} [c{0.03h} ; grid(n_tests,n_p) ; b{0.03h}]]
figi_t = plot(p_ylab, p_title, figsi_t..., p_xlab, layout=l, size=(450,300))
figi_t_name = dir_plt * "pval_gauss_equiv_truth_indep_true.pdf"
savefig(figi_t, figi_t_name)

p_title = shared_xlab("Reused comparison set (Approx. sampling)", bottom_margin=-3mm)
p_xlab = shared_xlab(L"$p$-value", top_margin=-1mm)
p_ylab = shared_ylab("Frequency", right_margin=6mm)
l = @layout [c{0.0001w} [c{0.03h} ; grid(n_tests,n_p) ; b{0.03h}]]
figr = plot(p_ylab, p_title, figsr..., p_xlab, layout=l, size=(450,300))
figr_name = dir_plt * "pval_gauss_equiv_truth_reuse_apx.pdf"
savefig(figr, figr_name)

p_title = shared_xlab("Reused comparison set (Exact sampling)", bottom_margin=-3mm)
p_xlab = shared_xlab(L"$p$-value", top_margin=-1mm)
p_ylab = shared_ylab("Frequency", right_margin=6mm)
l = @layout [c{0.0001w} [c{0.03h} ; grid(n_tests,n_p) ; b{0.03h}]]
figr_t = plot(p_ylab, p_title, figsr_t..., p_xlab, layout=l, size=(450,300))
figr_t_name = dir_plt * "pval_gauss_equiv_truth_reuse_true.pdf"
savefig(figr_t, figr_t_name)

println("Created $(figi_name) and $(figr_name)")
println("Created $(figi_t_name) and $(figr_t_name)")


# Synthetic experiment: permutation
# ---------------------------------

println("Creating permutation experiment plots")

# Set up experiment parameters
ns = GV_GAUSS_PER_n
n_n = length(ns)

ss = GV_GAUSS_PER_s
n_s = length(ss)

ds = GV_GAUSS_PER_d
n_d = length(ds)

# Read experiment results
n_tests = length(all_tests)
means = Array{Float64}(undef, n_tests, n_n, n_s, n_d)
sds = Array{Float64}(undef, n_tests, n_n, n_s, n_d)
@views @inbounds @threads for i in 1:n_d
    for j in 1:n_s
        for k in 1:n_n
            pvals = CSV.read(dir_out*"gauss_perm_S$(ds[i])_N1000_B100_n$(ns[k])_s$(ss[j]).csv", DataFrame)
            compute_rej_rate(pvals, all_tests, means[:,k,j,i], sds[:,k,j,i])
        end
    end
end

# Make plots
figs = Matrix{Any}(undef, n_s, n_d)
@views @inbounds for i in 1:n_d
    for j in 1:n_s
        xticks = i==n_d
        yticks = j==1
        ylab = j==1 ? L"d=%$(ds[i])" : ""
        title = i==1 ? L"s=%$(ss[j])" : ""
        figs[j,i] = plot_rej_rate(means[:,:,j,i], sds[:,:,j,i], all_tests_r, ns, 
                                  ylab=ylab, title=title, pl_xticks=xticks, pl_yticks=yticks)
    end
end
p_xlab = shared_xlab(L"n", top_margin=0mm)
p_ylab = shared_ylab("Reject rate", right_margin=4.5mm)
l = @layout [c{0.0001w} [a{0.03h} ; grid(n_d,n_s); b{0.03h}]]
fig = plot(p_ylab, leg,
                   figs...,
                   p_xlab, layout=l, size=(600,325))
fig_name = dir_plt * "gauss_equiv_perm_reuse.pdf"
savefig(fig, fig_name)
println("Created $(fig_name)")

@views @inbounds for i in 1:n_d
    for j in 1:n_s
        xticks = i==n_d
        yticks = j==1
        ylab = j==1 ? L"d=%$(ds[i])" : ""
        title = i==1 ? L"s=%$(ss[j])" : ""
        figs[j,i] = plot_rej_rate(means[:,:,j,i], sds[:,:,j,i], all_tests_i, ns, 
                                  ylab=ylab, title=title, pl_xticks=xticks, pl_yticks=yticks)
    end
end
p_xlab = shared_xlab(L"n", top_margin=0mm)
p_ylab = shared_ylab("Reject rate", right_margin=4.5mm)
l = @layout [c{0.0001w} [a{0.03h} ; grid(n_d,n_s); b{0.03h}]]
fig = plot(p_ylab, leg,
                   figs...,
                   p_xlab, layout=l, size=(600,325))
fig_name = dir_plt * "gauss_equiv_perm_indep.pdf"
savefig(fig, fig_name)
println("Created $(fig_name)")


# Synthetic experiment: number of randomizations
# ----------------------------------------------

println("Creating number of randomizations experiment plots")

# Set up experiment parameters
ps = GV_GAUSS_RES_p
n_p = length(ps)

ns = GV_GAUSS_RES_n
n_n = length(ns)

Bs = GV_GAUSS_RES_B
n_B = length(Bs)

p0_lims = (0, 0.05)
p099_lims = (0, 1)

# Read experiment results
n_tests = length(all_tests)
means = Array{Float64}(undef, n_tests, n_B, n_n, n_p)
sds = Array{Float64}(undef, n_tests, n_B, n_n, n_p)
@views @inbounds @threads for i in 1:n_p
    for j in 1:n_n
        for k in 1:n_B
            p = ps[i]
            n = ns[j]
            B = Bs[k]
            pvals = CSV.read(dir_out*"gauss_resamples_SO3_N1000_n$(n)_p$(p)_B$(B).csv", DataFrame)
            compute_rej_rate(pvals, all_tests, means[:,k,j,i], sds[:,k,j,i])
        end
    end
end

# Make plots
figsi = Matrix{Any}(undef, n_n, n_p)
figsr = Matrix{Any}(undef, n_n, n_p)
@views @inbounds for i in 1:n_p
    for j in 1:n_n
        title = i==1 ? L"$n=%$(ns[j])$" : ""
        ylab = j==1 ? L"$p=%$(ps[i])$" : ""
        pl_yticks = j==1
        ylims = i==1 ? p0_lims : p099_lims
        pl_xticks = i==2
        figsi[j,i] = plot_rej_rate(means[:,:,j,i], sds[:,:,j,i], all_tests_i, Bs, ylims=ylims,
                                   ylab=ylab, title=title, pl_xticks=pl_xticks, pl_yticks=pl_yticks, xrotation=30)
        figsr[j,i] = plot_rej_rate(means[:,:,j,i], sds[:,:,j,i], all_tests_r, Bs, ylims=ylims,
                                   ylab=ylab, title=title, pl_xticks=pl_xticks, pl_yticks=pl_yticks, xrotation=30)
    end
end

p_xlab = shared_xlab(L"B", top_margin=1mm)
p_title = shared_xlab("Independent comparison sets", top_margin=1mm, bottom_margin=-3mm)
p_ylab = shared_ylab("Reject rate", right_margin=4.5mm)
l = @layout [c{0.0001w} [a{0.05h} ; d{0.01h} ; grid(n_p,n_n) ; b{0.05h}]]
figi = plot(p_ylab, leg, p_title,
                   figsi...,
                   p_xlab, layout=l, size=(600,275))
figi_name = dir_plt * "gauss_equiv_resamples_indep.pdf"
savefig(figi, figi_name)

p_xlab = shared_xlab(L"B", top_margin=1mm)
p_title = shared_xlab("Reused comparison set", top_margin=1mm, bottom_margin=-3mm)
p_ylab = shared_ylab("Reject rate", right_margin=4.5mm)
l = @layout [c{0.0001w} [a{0.05h} ; d{0.01h} ; grid(n_p,n_n) ; b{0.05h}]]
figr = plot(p_ylab, leg, p_title,
                   figsr...,
                   p_xlab, layout=l, size=(600,275))
figr_name = dir_plt * "gauss_equiv_resamples_reuse.pdf"
savefig(figr, figr_name)
println("Created $(figi_name) and $(figr_name)")


# Synthetic experiment: non-equivariance in mean
# ----------------------------------------------

println("Creating non-equivariance in mean experiment plots")

# Set up experiment parameters
ps = GV_GAUSS_SEN_p
n_p = length(ps)

ss = GV_GAUSS_SEN_s
n_s = length(ss)

ns = GV_GAUSS_SEN_n
n_n = length(ns)

# Read experiment results
n_tests = length(all_tests)
means = Array{Float64}(undef, n_tests, n_n, n_s, n_p)
sds = Array{Float64}(undef, n_tests, n_n, n_s, n_p)
@views @inbounds @threads for i in 1:n_p
    for j in 1:n_s
        for k in 1:n_n
            pvals = CSV.read(dir_out*"gauss_sens_SO3_N1000_B100_p$(ps[i])_s$(ss[j])_n$(ns[k]).csv", DataFrame)
            compute_rej_rate(pvals, all_tests, means[:,k,j,i], sds[:,k,j,i])
        end
    end
end

# Make plots
figsi = Matrix{Any}(undef, n_s, n_p)
figsr = Matrix{Any}(undef, n_s, n_p)
@views @inbounds for i in 1:n_p
    for j in 1:n_s
        title = i==1 ? L"$s=%$(ss[j])$" : ""
        ylab = j==1 ? L"$p=%$(round(ps[i]/(2*π),digits=3))$" : ""
        pl_yticks = j==1
        pl_xticks = i==4
        figsi[j,i] = plot_rej_rate(means[:,:,j,i], sds[:,:,j,i], all_tests_i, ns,
                                   ylab=ylab, title=title, pl_xticks=pl_xticks, pl_yticks=pl_yticks)
        figsr[j,i] = plot_rej_rate(means[:,:,j,i], sds[:,:,j,i], all_tests_r, ns,
                                   ylab=ylab, title=title, pl_xticks=pl_xticks, pl_yticks=pl_yticks)
    end
end

p_xlab = shared_xlab(L"B", top_margin=1mm)
p_ylab = shared_ylab("Reject rate", right_margin=4.5mm)
l = @layout [c{0.0001w} [a{0.01h} ; grid(n_p,n_s) ; b{0.05h}]]
figi = plot(p_ylab, leg,
                   figsi...,
                   p_xlab, layout=l, size=(600,425))
figi_name = dir_plt * "gauss_equiv_sensitivity_indep.pdf"
savefig(figi, figi_name)

p_xlab = shared_xlab(L"B", top_margin=1mm)
p_ylab = shared_ylab("Reject rate", right_margin=4.5mm)
l = @layout [c{0.0001w} [a{0.01h} ; grid(n_p,n_s) ; b{0.05h}]]
figr = plot(p_ylab, leg,
                   figsr...,
                   p_xlab, layout=l, size=(600,425))
figr_name = dir_plt * "gauss_equiv_sensitivity_reuse.pdf"
savefig(figr, figr_name)
println("Created $(figi_name) and $(figr_name)")


# MNIST experiment
# ----------------

println("Creating MNIST plots")

# Read experiment results
str_9 = ["w9", "wo9"]
str_aug = ["naug", "aug"]

# Make plots
x = 0:3:30

# Plot unaugmented results
ps = Matrix{Any}(undef, 2, 2)
@inbounds @views for i in 1:length(str_9)
    for j in 1:length(str_aug)
        nine = str_9[i]
        aug = str_aug[j]
        df = CSV.read(dir_out*"MNIST_N1000_n100_B100_" * aug * "_" * nine * ".csv", DataFrame)
        n = size(df, 1)
        
        legend = i==1 && j==2 ? :topright : false
        ylab = j==1 ? (i==1 ? "9 included" : "9 excluded") : ""
        title = i==1 ? (j==1 ? "Without DA" : "With DA") : ""
        
        mn = mean.(eachcol(df[:,Regex("reuse_.*_FUSE_p\$")] .< α))
        ps[j,i] = plot(x, mn, linecolor=colsDict[GV_FUSE], linealpha=0.7, ylab=ylab, title=title, label="R-"*GV_FUSE,
                       xlims=(minimum(x),maximum(x)), ylims=(0,1), grid=true, legend=legend)
        sd = sqrt.(mn .* (1 .- mn) / n)
        plot!(x, max.(0,mn.-sd), fillrange=min.(1,mn.+sd), label=:none, fillalpha=0.2, linealpha=0, fillcolor=colsDict[GV_FUSE])
        
        mn = mean.(eachcol(df[:,Regex("indep_.*_FUSE_p\$")] .< α))
        plot!(x, mn, linecolor=colsDict[GV_FUSE], linealpha=0.7, linestyle=:dot, label="I-"*GV_FUSE)
        sd = sqrt.(mn .* (1 .- mn) / n)
        plot!(x, max.(0,mn.-sd), fillrange=min.(1,mn.+sd), label=:none, fillalpha=0.2, linealpha=0, fillcolor=colsDict[GV_FUSE])

        mn = mean.(eachcol(df[:,Regex("reuse_.*_SK_p\$")] .< α))
        plot!(x, mn, linecolor=colsDict[GV_SK], linealpha=0.7, label="R-"*GV_SK)
        sd = sqrt.(mn .* (1 .- mn) / n)
        plot!(x, max.(0,mn.-sd), fillrange=min.(1,mn.+sd), label=:none, fillalpha=0.2, linealpha=0, fillcolor=colsDict[GV_SK])
        
        mn = mean.(eachcol(df[:,Regex("indep_.*_SK_p\$")] .< α))
        plot!(x, mn, linecolor=colsDict[GV_SK], linealpha=0.7, linestyle=:dot, label="I-"*GV_SK)
        sd = sqrt.(mn .* (1 .- mn) / n)
        plot!(x, max.(0,mn.-sd), fillrange=min.(1,mn.+sd), label=:none, fillalpha=0.2, linealpha=0, fillcolor=colsDict[GV_SK])
        
        if i==1
            plot!(xticks = (xticks(ps[j,i])[1][1],""))
        end
        if j==2
            plot!(yticks = (yticks(ps[j,i])[1][1],""))
        end
    end
end
p_xlab = shared_xlab("Epoch", top_margin=-1mm)
p_ylab = shared_ylab("Reject rate", right_margin=1mm)
l = @layout [c{0.0001w} grid(2,2) ; e{0.0001w,0.01h} b{0.01h}]
fig = plot(p_ylab, ps...,
           ep, p_xlab, layout=l, size=(350,250))
fig_name = dir_plt * "MNIST.pdf"
savefig(fig, fig_name)
println("Created $(fig_name)")


# Invariance experiment
# ---------------------

println("Creating invariance plots")

# Set up experiment parameters
ps = GV_INV_p
n_p = length(ps)
title_labs = [L"p=\pi/2",L"p=\pi",L"p=3\pi/4",L"p=2\pi"]

ns = GV_INV_n
n_n = length(ns)

# Read experiment results
n_tests = length(all_tests)
means = Array{Float64}(undef, n_tests, n_n, n_p)
sds = Array{Float64}(undef, n_tests, n_n, n_p)
@views @inbounds @threads for i in 1:n_p
    for j in 1:n_n
        pvals = CSV.read(dir_out*"invariance_N1000_n$(ns[j])_p$(ps[i])_B100.csv", DataFrame)
        compute_rej_rate(pvals, all_tests, means[:,j,i], sds[:,j,i])
    end
end

# Make plots
figs = Vector{Any}(undef, n_p)
@views @inbounds for i in 1:n_p
    yticks = i==1
    figs[i] = plot_rej_rate(means[:,:,i], sds[:,:,i], all_tests, ns, title=title_labs[i], pl_xticks=true, pl_yticks=yticks)
end
p_xlab = shared_xlab(L"n", top_margin=2mm)
p_ylab = shared_ylab("Reject rate", right_margin=4.5mm)
l = @layout [c{0.0001w} [a{0.1h} ; grid(1,n_p) ; b{0.05h}]]
fig = plot(p_ylab, leg, figs..., p_xlab, layout=l, size=(600,200))
fig_name = dir_plt * "invariance.pdf"
savefig(fig, fig_name)
println("Created $(fig_name)")

# p-value plots
n_tests = length(test_labs)
figs = Matrix{Any}(undef, n_p, n_tests)
@views @inbounds @threads for i in 1:n_p
    pvals = CSV.read(dir_out*"invariance_N1000_n100_p$(ps[i])_B100.csv", DataFrame)
    for j in 1:n_tests
        last_row = j==n_tests
        title = j==1 ? title_labs[i] : ""
        ylab = i==1 ? test_labs[j] : ""
        figs[i,j] = hist(pvals, test_labs[j], ylab, title=title, last_row=last_row)
    end
end
ep = shared_xlab("")
p_xlab = shared_xlab(L"$p$-value", top_margin=0mm)
p_ylab = shared_ylab("Frequency", right_margin=6mm)
l = @layout [c{0.0001w} [c{0.03h} ; grid(n_tests,n_p) ; b{0.03h}]]
fig = plot(p_ylab, ep, figs..., p_xlab, layout=l, size=(450,350))
fig_name = dir_plt * "pval_invariance.pdf"
savefig(fig, fig_name)
println("Created $(fig_name)")


# LHC data figure
# ---------------

Random.seed!(seed)

# Read in data
fid = h5open(dir_out_dat * "LHC.h5", "r")
LHC = copy(read(fid["data"]))
close(fid)
n_LHC = size(LHC, 2)

# Sample data
n = 500
inds = sample(1:n_LHC, n, replace=false)
x = LHC[[1,2,3,4], inds]

# Plot the data
m = maximum(abs.(x)) + 100
lims = (-m,m)
p = @views scatter(x[1,:],x[2,:], markersize=3, markerstrokewidth=0, markeralpha=0.5, lims=lims, xlab=L"p_1", markercolor="darkorange2",
                   label="1st LC", title="Leading Constituent momenta", titlefontsize=9)
@views scatter!(x[3,:],x[4,:], markersize=3, markerstrokewidth=0, markeralpha=0.4, ylab=L"p_2", label="2nd LC", markercolor=1)
fig = plot(p, size=(250,250), dpi=300, legend=true)
fig_name = dir_plt * "LHC_data.pdf"
savefig(fig, fig_name)
println("Created $(fig_name)")
# geostrophic balance and conservation of energy
from firedrake import *
import matplotlib.pyplot as plt
# import numpy as np
from split_initializations import *
import math

H = 10
Dt = 0.0005
# Dt = 0.001
dt = Constant(Dt)

n = 10
# n = 10
Lx = 5000  # in km                           # Zonal length
Ly = 4330  # in km                           # Meridonal length

mesh = PeriodicRectangleMesh(n, n, Lx, Ly)
n_vec = FacetNormal(mesh)
CR = VectorFunctionSpace(mesh, "CR", 1)  # BDM 1 or RT 1
RT = FunctionSpace(mesh, "BDM", 1)  # BDM 1 or RT 1
DG = FunctionSpace(mesh, "DG", 0)
CG = FunctionSpace(mesh, "CG", 1)

### Select initial conditions!!!
x = SpatialCoordinate(mesh)  # define (x[0], x[1])

## 1) vortex pair interaction:
# ~ Dn0 = Function(VD).interpolate(vortex_pair_elevation(x[0],x[1],H,DeltaH,Lx,Ly))
# ~ hn0 = Function(Vh).project(Dn0)
# hn0 = Function(CG).interpolate(vortex_pair_elevation(x[0],x[1],H,DeltaH,Lx,Ly))
# Dn0 = Function(DG).project(hn0)

## 2) shear flow:
# hn0 = Function(Vh).interpolate(shear_elevation(x[0],x[1],H,DeltaH,Lx,Ly))
# Dn0 = Function(VD).project(hn0)
#Dn0 = Function(VD).interpolate(shear_elevation(x[0],x[1],H,DeltaH,Lx,Ly))
#hn0 = Function(Vh).project(Dn0)

## 3) single vortex tc:
#Dn0 = Function(DG).interpolate(vortex_single_elevation_pos(x[0], x[1], H, DeltaH, Lx, Ly))
#hn0 = Function(CG).project(Dn0)
hn0 = Function(CG).interpolate(vortex_single_elevation(x[0],x[1],H,DeltaH,Lx,Ly))
Dn0 = Function(DG).project(hn0)
#Dn0 = Function(VD).interpolate(vortex_single_elevation_pos(x[0],x[1],H,DeltaH,Lx,Ly))
#hn0 = Function(Vh).project(Dn0)
un0 = project(perp(grad(hn0)), RT)
# un0 *= 0
un0 *= g/f
# print(f)
# print(g)
# trisurf(Dn0)
# quiver(un0)
# plt.show()


# Define weak problem:
W = RT * DG   # Mixed space for velocity and depth

U = Function(W) # U = TrialFunctions(W)
u, D = U.split()
D.assign(Dn0)
u.assign(un0)

# define velocity and depth increment
dU_trial = TrialFunction(W)
du_trial, dh_trial = split(dU_trial)
dU = Function(W)


# define test functions!
v, phi = TestFunctions(W)

U1 = Function(W)
u1, D1 = split(U1)
U2 = Function(W)  
u2, D2 = split(U2) 

def mass_function(du, dh):
    lhs = inner(v, du) * dx + phi * dh * dx
    return lhs
    
def form_function(u, D):
    rhs = (dt * g * D * div(v) * dx 
    - dt * H * phi * div(u) * dx
    - dt * inner(v , f * perp(u)) * dx)
    return rhs  

lhs = mass_function(du_trial, dh_trial)
rhs0 = form_function(u, D)
rhs1 = form_function(u1, D1)
rhs2 = form_function(u2, D2)
          
# params = {'mat_type': 'aij',
#           'ksp_type': 'preonly',
#           'pc_type': 'lu'}

params = {'ksp_type': 'preonly', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}

uh_problem0 = LinearVariationalProblem(lhs, rhs0, dU)
uh_solver0 = LinearVariationalSolver(uh_problem0,
                                       solver_parameters=params)
uh_problem1 = LinearVariationalProblem(lhs, rhs1, dU)
uh_solver1 = LinearVariationalSolver(uh_problem1,
                                       solver_parameters=params)
uh_problem2 = LinearVariationalProblem(lhs, rhs2, dU)
uh_solver2 = LinearVariationalSolver(uh_problem2,
                                       solver_parameters=params)

# check geostrophic balance here, calculate u_t
ut_trial = TrialFunction(RT)
v4 = TestFunction(RT)
ut = Function(RT)
eqn5 = inner(v4 , ut_trial) * dx
eqn6 = (div(v4) * g * D * dx - inner(v4 , f * perp(u)) * dx)
params = {'ksp_type': 'preonly', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}
ut_problem = LinearVariationalProblem(eqn5, eqn6, ut)
ut_solver = LinearVariationalSolver(ut_problem,
                                       solver_parameters=params)

q = (f - div(perp(u))) / D
q_out = Function(CG, name="Vorticity").project(q)
u.rename("Velocity")
D.rename("Depth")

# e_tot_0 = assemble(0.5 * inner(un0, H * un0) * dx + 0.5 * g * (Dn0**2) * dx)  # define total energy at each step
e_tot_0 = assemble(0.5 * inner(un0, un0) * H * dx + 0.5 * g * (Dn0 * Dn0) * dx)  # define total energy at each step
all_e_tot = []
ens_0 = assemble(q**2 * D * dx)  # define enstrophy at each time step
all_ens = []

# name = "lsw_rk_BDM"
# ufile = File("output/" + name + ".pvd")

tmax = 4  # days
# tmax = 100
t = 0.
dt1 = 1
dumpfreq = 10
dumpcount = dumpfreq

norm_err = []
# norm_err2 = []
# uerrors = []
# herrors = []
# u0norm = norm(un0, norm_type="L2")
# h0norm = norm(Dn0, norm_type="L2")
# norm0 = norm(un0) + norm(Dn0)


def dump():
    global dumpcount
    dumpcount += 1
    if (dumpcount > dumpfreq):
        # ufile.write(u, D, q_out, time=t)
        dumpcount -= dumpfreq


dump()

while t < tmax / Dt - dt1 / 2:
    t += dt1
    print("t= ", t * Dt)  
    # proj_solver.solve() 
    # r_solver.solve()
    uh_solver0.solve()
    U1.assign(U + dU)
    uh_solver1.solve()
    U2.assign(0.75*U + 0.25*(U1 + dU))
    uh_solver2.solve()
    U.assign((1.0/3.0)*U + (2.0/3.0)*(U2 + dU))
    ut_solver.solve()
    dump()
    #convergence test
    # uerr = errornorm(u, un0, norm_type="L2")
    # herr = errornorm(D, Dn0, norm_type="L2")
    # uerrors.append(uerr)
    # herrors.append(herr)
    # energy conservation
    # e_tot_t = assemble(0.5 * inner(u, H * u) * dx + 0.5 * g * (D**2) * dx)
    e_tot_t = assemble(0.5 * inner(u, u) * H * dx + 0.5 * g * (D * D) * dx)

    all_e_tot.append(abs(e_tot_t / e_tot_0 - 1))
    #geo_balance test
    geo_t = norm(ut, norm_type="L2")
    # geo_t = norm(f * perp(u) + g * grad(D))
    norm_err.append(geo_t)
    # norm1 = norm(u) + norm(D)
    # norm_err2.append(norm1 - norm0)
    # norm_err_t = abs(Norm(u, D) - Norm0)
    # norm_err.append(norm_err_t)
    print(e_tot_t / e_tot_0 - 1, 'Energy')


plt.figure()    
plt.xticks([0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000], [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
plt.plot(all_e_tot)
plt.xlabel("time/days")
plt.ylabel("total energy")

plt.figure()
plt.xticks([0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000], [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
# plt.plot(norm_err/norm_err[0]-1)
plt.plot(norm_err)
plt.xlabel('time/days')
plt.ylabel('numerical solution of $u_t$')

plt.show()

# plt.figure()
# plt.xticks([0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000], [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
# plt.plot(norm_err2)
# plt.xlabel('time/days')
# plt.ylabel('norm error')

# plt.show()


#plot u error and h error
# plt.figure()
# plt.xticks([0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000], [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
# plt.plot(uerrors)
# plt.xlabel('time/days')
# plt.ylabel('normalized error of $u$')

# plt.show()

# plt.figure()
# plt.xticks([0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000], [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
# plt.plot(herrors)
# plt.xlabel('time/days')
# plt.ylabel('normalized error of $h$')

# plt.show()
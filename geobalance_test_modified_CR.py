from firedrake import *
import matplotlib.pyplot as plt
from split_initializations import *
# from numpy import *
# from projector import *

H = 10
Dt = 0.001 #works with n=50
# Dt = 0.00025

dt = Constant(Dt)

n = 10
Lx = 5000  # in km                           # Zonal length
Ly = 4330  # in km                           # Meridonal length

mesh = PeriodicRectangleMesh(n, n, Lx, Ly)
n_vec = FacetNormal(mesh)
CR = VectorFunctionSpace(mesh, "CR", 1)  # BDM 1 or RT 1
BDM = FunctionSpace(mesh, "BDM", 1)
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
un0 = project(perp(grad(hn0)), CR)
# un0 *= 0
un0 *= g/f
# print(f)
# print(g)
# trisurf(Dn0)
# quiver(un0)
# plt.show()


# Define weak problem:
W = CR * DG   # Mixed space for velocity and depth

U = Function(W) # U = TrialFunctions(W)
u, D = U.split()
D.assign(Dn0)
u.assign(un0)

#def projection problem here
u_trial = TrialFunction(BDM)
v2 = TestFunction(BDM)
u_hat = Function(BDM)

eqn1 = avg(inner(v2 , n_vec) * inner(u_trial , n_vec)) * dS
eqn2 = avg(inner(v2 , n_vec) * inner(u , n_vec)) * dS

proj_problem = LinearVariationalProblem(eqn1, eqn2, u_hat)
params = {'ksp_type': 'preonly', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}
proj_solver = LinearVariationalSolver(proj_problem,
                                       solver_parameters=params)
# proj_solver.solve()

# from IPython import embed; embed()

# define a solver for r
r_trial = TrialFunction(BDM)
v3 = TestFunction(BDM)
r = Function(BDM)
eqn3 = avg(inner(v3, n_vec) * inner(r_trial, n_vec)) * dS

# from IPython import embed; embed()
eqn4 = inner(v3, f * perp(u_hat)) * dx
params = {'ksp_type': 'preonly', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}
r_problem = LinearVariationalProblem(eqn3, eqn4, r)
r_solver = LinearVariationalSolver(r_problem,
                                    solver_parameters=params)
# r_hat = r_solver.solve()
# r_solver.solve()

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
    # proj_u = proj(u)
    # r_solver.solve()
    rhs = (dt * g * D * div(v) * dx 
    - dt * H * phi * div(u) * dx
    - dt * avg(inner(r, n_vec) * inner(v, n_vec)) * dS)
    return rhs  

# from IPython import embed; embed()

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
ut_trial = TrialFunction(CR)
v4 = TestFunction(CR)
ut = Function(CR)
eqn5 = inner(v4 , ut_trial) * dx
# eqn6 = div(v4) * g * D * dx - inner(v4 , f * perp(u)) * dx
eqn6 = div(v4) * g * D * dx - avg(inner(r, n_vec) * inner(v4, n_vec)) * dS
params = {'ksp_type': 'preonly', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}
ut_problem = LinearVariationalProblem(eqn5, eqn6, ut)
ut_solver = LinearVariationalSolver(ut_problem,
                                       solver_parameters=params)



u.rename("Velocity")
D.rename("Depth")

e_tot_0 = assemble(0.5 * inner(u**2, H) * dx + 0.5 * g * (D ** 2) * dx)  # define total energy at each step
all_e_tot = []


name = "geobalance_test_CR_modified"
ufile = File("output/" + name + ".pvd")

tmax = 4  # days
t = 0.
dt1 = 1
dumpfreq = 10
dumpcount = dumpfreq


# from IPython import embed; embed()
#calculate norm here
Norm0 = norm(un0) + norm(Dn0)

# def Norm(u, D):
#     Norm = norm(u) + norm(D)
#     return Norm



norm_err = []

def dump():
    global dumpcount
    dumpcount += 1
    if (dumpcount > dumpfreq):
        # ufile.write(u, D, time=t)
        dumpcount -= dumpfreq


dump()

while t < tmax / Dt - dt1 / 2:
    t += dt1
    print("t= ", t * Dt)  
    proj_solver.solve() 
    r_solver.solve()
    uh_solver0.solve()
    U1.assign(U + dU)
    uh_solver1.solve()
    U2.assign(0.75*U + 0.25*(U1 + dU))
    uh_solver2.solve()
    U.assign((1.0/3.0)*U + (2.0/3.0)*(U2 + dU))
    ut_solver.solve()
    dump()
    e_tot_t = assemble(0.5 * inner(u, H * u) * dx + 0.5 * g * (D ** 2) * dx)
    all_e_tot.append(e_tot_t / e_tot_0 - 1)
    geo_t = norm(ut, norm_type="L2")
    norm_err.append(geo_t)
    # norm_err_t = abs(Norm(u, D) - Norm0)
    # norm_err.append(norm_err_t)
    print(e_tot_t / e_tot_0 - 1, 'Energy')


# plt.figure()    
# plt.plot(all_e_tot)
# plt.xlabel("time/days")
# plt.ylabel("total energy")

plt.figure()
plt.xticks([0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000], [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
plt.plot(norm_err/norm_err[0])
plt.xlabel('time/days')
plt.ylabel('L2 norm of $u_t$')

plt.show()
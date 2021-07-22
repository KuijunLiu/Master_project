from firedrake import *
import matplotlib.pyplot as plt
from split_initializations import *

H = 10
Dt = 0.001
dt = Constant(Dt)

n = 50
Lx = 5000  # in km                           # Zonal length
Ly = 4330  # in km                           # Meridonal length

mesh = PeriodicRectangleMesh(n, n, Lx, Ly)
RT = FunctionSpace(mesh, "RT", 1)  # BDM 1 or RT 1
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
Dn0 = Function(DG).interpolate(vortex_single_elevation_pos(x[0], x[1], H, DeltaH, Lx, Ly))
hn0 = Function(CG).project(Dn0)
# ~ hn0 = Function(Vh).interpolate(vortex_single_elevation_pos(x[0],x[1],H,DeltaH,Lx,Ly)) 
# ~ Dn0 = Function(VD).project(hn0)
#Dn0 = Function(VD).interpolate(vortex_single_elevation_pos(x[0],x[1],H,DeltaH,Lx,Ly))
#hn0 = Function(Vh).project(Dn0)
un0 = project(perp(grad(hn0)), RT)
un0 *= g/f
un0 *= 0




# Define weak problem:
W = RT * DG   # Mixed space for velocity and depth

U = Function(W) # U = TrialFunctions(W)
u, D = U.split()
D.assign(Dn0)
u.project(un0)

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
    rhs = dt * g * D * div(v) * dx - dt * H * phi * div(u) * dx
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


e_tot_0 = assemble(0.5 * inner(u**2, D) * dx + 0.5 * g * (D ** 2) * dx)  # define total energy at each step
all_e_tot = []
#ens_0 = assemble(q**2 * D * dx)  # define enstrophy at each time step
#all_ens = []

name = "lsw_rk"
ufile = File("output/" + name + ".pvd")

tmax = 4  # days
t = 0.
dt1 = 1
dumpfreq = 10
dumpcount = dumpfreq


def dump():
    global dumpcount
    dumpcount += 1
    if (dumpcount > dumpfreq):
        ufile.write(u, D, time=t)
        dumpcount -= dumpfreq


#dump()

while t < tmax / Dt - dt1 / 2:
    t += dt1
    print("t= ", t * Dt)   
    uh_solver0.solve()
    U1.assign(U + dU)
    uh_solver1.solve()
    U2.assign(0.75*U + 0.25*(U1 + dU))
    uh_solver2.solve()
    U.assign((1.0/3.0)*U + (2.0/3.0)*(U2 + dU))
    ufile.write(u, D, time=t)
    e_tot_t = assemble(0.5 * inner(u, D * u) * dx + 0.5 * g * (D ** 2) * dx)
    all_e_tot.append(e_tot_t / e_tot_0 - 1)
    #ens_t = assemble(q**2 * D * dx)
    #all_ens.append(ens_t/ens_0 - 1)
    print(e_tot_t / e_tot_0 - 1, 'Energy')
    #print(ens_t / ens_0 - 1, 'Enstrophy')
    

fig1 = plt.plot(all_e_tot)
plt.show()
#fig2 = plt.plot(all_ens)
#plt.show()
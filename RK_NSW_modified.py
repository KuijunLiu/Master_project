from firedrake import *
import matplotlib.pyplot as plt
from split_initializations import *

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
hn0 = Function(CG).interpolate(vortex_pair_elevation(x[0],x[1],H,DeltaH,Lx,Ly))
Dn0 = Function(DG).project(hn0)

## 2) shear flow:
# hn0 = Function(Vh).interpolate(shear_elevation(x[0],x[1],H,DeltaH,Lx,Ly))
# Dn0 = Function(VD).project(hn0)
#Dn0 = Function(VD).interpolate(shear_elevation(x[0],x[1],H,DeltaH,Lx,Ly))
#hn0 = Function(Vh).project(Dn0)

## 3) single vortex tc:
#Dn0 = Function(DG).interpolate(vortex_single_elevation_pos(x[0], x[1], H, DeltaH, Lx, Ly))
#hn0 = Function(CG).project(Dn0)
# ~ hn0 = Function(Vh).interpolate(vortex_single_elevation_pos(x[0],x[1],H,DeltaH,Lx,Ly)) 
# ~ Dn0 = Function(VD).project(hn0)
#Dn0 = Function(VD).interpolate(vortex_single_elevation_pos(x[0],x[1],H,DeltaH,Lx,Ly))
#hn0 = Function(Vh).project(Dn0)
un0 = project(perp(grad(hn0)), RT)
un0 *= g/f





# Define weak problem:
W = RT * DG * CG * RT  # Mixed space for velocity, depth, potential vorticity and volume flux
#L = RT * DG
U = Function(W)
u, D, q, F = U.split()
D.assign(Dn0)
u.project(un0)

# define time derivative
dU = Function(W)
du, dh, q, F = split(dU)

# define test functions!
v, phi, r, tF = TestFunctions(W)

uh_eqn1 = (inner(v, dt * du) * dx + dt * inner(v , q * perp(F)) * dx #change q_h and F_h to q_n and F_n
          - dt * div(v) * (g * D + 0.5 * u**2) * dx 
          + phi * dt * dh * dx 
          + dt * phi * div(F) * dx   #dot(,)       
          + inner(tF, F - D * u) * dx
          + inner(r, q * D - f) * dx + inner(perp(grad(r)) , u) * dx
          ) 
 
U1 = Function(W)
u1, D1, q1, F1 = split(U1)
U2 = Function(W)  
u2, D2, q2, F2 = split(U2) 
uh_eqn2 = replace(uh_eqn1, {u:u1, D:D1, q:q1, F:F1})  
uh_eqn3 = replace(uh_eqn2, {u:u2, D:D2, q:q2, F:F2})     
          
params = {'mat_type': 'aij',
          'ksp_type': 'preonly',
          'pc_type': 'lu'}
uh_problem1 = NonlinearVariationalProblem(uh_eqn1, dU)
uh_solver1 = NonlinearVariationalSolver(uh_problem1,
                                       solver_parameters=params)
uh_problem2 = NonlinearVariationalProblem(uh_eqn2, dU)
uh_solver2 = NonlinearVariationalSolver(uh_problem2,
                                       solver_parameters=params)
uh_problem3 = NonlinearVariationalProblem(uh_eqn3, dU)
uh_solver3 = NonlinearVariationalSolver(uh_problem3,
                                       solver_parameters=params)

#uh_solver1.solve()
# u1 = (u + du); D1 = (D + dh)
#uh_solver2.solve()
# u2 = (0.75*u + 0.25*(u1 + du))
# D2 = (0.75*D + 0.25*(D1 + dh))
#uh_solver3.solve()
# u = ((1.0/3.0)*u + (2.0/3.0)*(u2 + du))
# D = ((1.0/3.0)*D + (2.0/3.0)*(D2 + dh))
#u.rename('Velocity')
#D.rename('Depth')
#q.project(q2)
#q.rename('Vorticity')

e_tot_0 = assemble(0.5 * inner(u**2, D) * dx + 0.5 * g * (D ** 2) * dx)  # define total energy at each step
all_e_tot = []
ens_0 = assemble(q**2 * D * dx)  # define enstrophy at each time step
all_ens = []

name = "nl_sw"
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
        ufile.write(u, D, q, time=t)
        dumpcount -= dumpfreq


#dump()

while t < tmax / Dt - dt1 / 2:
    t += dt1
    print("t= ", t * Dt)   
    uh_solver1.solve()
    U1.assign(U + dU)
    uh_solver2.solve()
    U2.assign(0.75*U + 0.25*(U1 + dU))
    uh_solver3.solve()
    U.assign((1.0/3.0)*U + (2.0/3.0)*(U2 + dU))
    ufile.write(u, D, q, time=t)
    e_tot_t = assemble(0.5 * inner(u, D * u) * dx + 0.5 * g * (D ** 2) * dx)
    all_e_tot.append(e_tot_t / e_tot_0 - 1)
    ens_t = assemble(q**2 * D * dx)
    all_ens.append(ens_t/ens_0 - 1)
    print(e_tot_t / e_tot_0 - 1, 'Energy')
    print(ens_t / ens_0 - 1, 'Enstrophy')
    

fig1 = plt.plot(all_e_tot)
plt.show()
fig2 = plt.plot(all_ens)
plt.show()

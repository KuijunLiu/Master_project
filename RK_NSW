from firedrake import *
import matplotlib.pyplot as plt
from split_initializations import *

Dt = 0.01
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
U_n = Function(W)
u_n, D_n, q_n, F_n = U_n.split()
D_n.assign(Dn0)
u_n.project(un0)

U_p = Function(W)
u_p, D_p, q_h, F_h = split(U_p)
U_p.assign(U_n)

U_p1 = Function(W)
u_p1, D_p1, q_h1, F_h1 = split(U_p1)
U_p1.assign(U_p)

U_p2 = Function(W)
u_p2, D_p2, q_h2, F_h2 = split(U_p2)
U_p2.assign(U_p1)

# define test functions!
v, phi, r, tF = TestFunctions(W)

h_h = 0.5 * (D_p + D_n)
u_h = 0.5 * (u_p + u_n)
#make some changes here, re-define F_h, q_h
#F_h = 0.5 * (F_p + F_n)
#q_h = 0.5 * (q_p + q_n)
uh_eqn = (inner(v, u_p - u_n) * dx + dt * inner(v , q_h * perp(F_h)) * dx #change q_h and F_h to q_n and F_n
          - dt * div(v) * (g * h_h + 0.5 * u_h**2) * dx 
          + phi * (D_p - D_n) * dx 
          + dt * phi * div(F_h) * dx   #dot(,)       
          + inner(tF, F_h - h_h * u_h) * dx
          + inner(r, q_h * h_h - f) * dx + inner(perp(grad(r)) , u_h) * dx
          ) 
          
uh_problem = NonlinearVariationalProblem(uh_eqn, U_p)
params = {'mat_type': 'aij',
          'ksp_type': 'preonly',
          'pc_type': 'lu'}
uh_solver1 = NonlinearVariationalSolver(uh_problem,
                                       solver_parameters=params)
uh_solver1.solve()

h_h1 = 0.5 * (D_p1 + D_p)
u_h1 = 0.5 * (u_p1 + u_p)
#make some changes here, re-define F_h, q_h
#F_h1 = 0.5 * (F_p1 + F_p)
#q_h1 = 0.5 * (q_p1 + q_p)
uh_eqn1 = (inner(v, 4 * u_p1 - 3 * u_n - u_p) * dx + dt * inner(v , q_h1 * perp(F_h1)) * dx #change q_h and F_h to q_n and F_n
          - dt * div(v) * (g * h_h1 + 0.5 * u_h1**2) * dx 
          + phi * (4 * D_p1 - 3 * D_n - D_p) * dx 
          + dt * phi * div(F_h1) * dx   #dot(,)       
          + inner(tF, F_h1 - h_h1 * u_h1) * dx
          + inner(r, q_h1 * h_h1 - f) * dx + inner(perp(grad(r)) , u_h1) * dx
          ) 
          
uh_problem1 = NonlinearVariationalProblem(uh_eqn1, U_p1)
params = {'mat_type': 'aij',
          'ksp_type': 'preonly',
          'pc_type': 'lu'}
uh_solver2 = NonlinearVariationalSolver(uh_problem1,
                                       solver_parameters=params)
uh_solver2.solve()


h_h2 = 0.5 * (D_p2 + D_p1)
u_h2 = 0.5 * (u_p2 + u_p1)
#make some changes here, re-define F_h, q_h
#F_h2 = 0.5 * (F_p2 + F_p1)
#q_h2 = 0.5 * (q_p2 + q_p1)
uh_eqn2 = (inner(v, 3 * u_p2 - u_n - 2 * u_p1) * dx + dt * inner(v , q_h2 * perp(F_h2)) * dx #change q_h and F_h to q_n and F_n
          - dt * div(v) * (g * h_h2 + 0.5 * u_h2**2) * dx 
          + phi * (3 * D_p2 - D_n - 2 * D_p1) * dx 
          + dt * phi * div(F_h2) * dx   #dot(,)       
          + inner(tF, F_h2 - h_h2 * u_h2) * dx
          + inner(r, q_h2 * h_h2 - f) * dx + inner(perp(grad(r)) , u_h2) * dx
          ) 
          
uh_problem2 = NonlinearVariationalProblem(uh_eqn2, U_p2)
params = {'mat_type': 'aij',
          'ksp_type': 'preonly',
          'pc_type': 'lu'}
uh_solver3 = NonlinearVariationalSolver(uh_problem2,
                                       solver_parameters=params)
uh_solver3.solve()


u_n.rename('Velocity')
D_n.rename('Depth')
q_n.project(q_h2)
q_n.rename('Vorticity')

e_tot_0 = assemble(0.5 * inner(u_n**2, D_n) * dx + 0.5 * g * (D_n ** 2) * dx)  # define total energy at each step
all_e_tot = []
ens_0 = assemble(q_h2**2 * D_n * dx)  # define enstrophy at each time step
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
        ufile.write(u_n, D_n, q_n, time=t)
        dumpcount -= dumpfreq


dump()

while t < tmax / Dt - dt1 / 2:
    t += dt1
    print("t= ", t * Dt)
    uh_solver1.solve()
    uh_solver2.solve()
    uh_solver3.solve()
    U_n.assign(U_p2)
    ufile.write(u_n, D_n, q_n, time=t)
    e_tot_t = assemble(0.5 * inner(u_n, D_n * u_n) * dx + 0.5 * g * (D_n ** 2) * dx)
    all_e_tot.append(e_tot_t / e_tot_0 - 1)
    ens_t = assemble(q_n**2 * D_n * dx)
    all_ens.append(ens_t/ens_0 - 1)
    print(e_tot_t / e_tot_0 - 1, 'Energy')
    print(ens_t / ens_0 - 1, 'Enstrophy')
    

fig1 = plt.plot(all_e_tot)
plt.show()
fig2 = plt.plot(all_ens)
plt.show()
    
    
  

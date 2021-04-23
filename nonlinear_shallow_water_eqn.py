from firedrake import *

Dt = 0.1
dt = Constant(Dt)

n = 50
Lx = 5000  # in km                           # Zonal length
Ly = 4330  # in km                           # Meridonal length

mesh = PeriodicRectangleMesh(n, n, Lx, Ly)
RT = FunctionSpace(mesh, "RT", 1)  # BDM 1 or RT 1
DG = FunctionSpace(mesh, "DG", 0)

## Initialization:

H = 10  # in km
DeltaH = 0.075
theta = 25. / 360. * 2. * pi
f = 2. * sin(theta) * 2. * pi  # in 1/days # - f * inner(p, perp(u_h)) * dx
g = 9.81 * (24. * 3600.) ** 2. / 1000.  # in  km/days^2


def vortex_single_elevation_pos(x, y, H, DeltaH, Lx, Ly):
    o = 0.1
    xc1 = (1. / 2. - o) * Lx
    yc1 = (1. / 2. - o) * Ly
    xc2 = (1. / 2. + o) * Lx
    yc2 = (1. / 2. + o) * Ly
    sx = 1.5 * Lx / 20.
    sy = 1.5 * Ly / 20.

    xp1 = Lx * sin(pi * (x - xc1) / Lx) / sx / pi
    yp1 = Ly * sin(pi * (y - yc1) / Ly) / sy / pi
    xp2 = 0 * Lx * sin(pi * (x - xc2) / Lx) / sx / pi
    yp2 = 0 * Ly * sin(pi * (y - yc2) / Ly) / sy / pi

    h = H + DeltaH * (exp(-1. / 2. * (xp1 ** 2. + yp1 ** 2.)) + exp(
        -1. / 2. * (xp2 ** 2. + yp2 ** 2.)) - 4. * pi * sx * sy / Lx / Ly)
    return h


x = SpatialCoordinate(mesh)  # define (x[0], x[1])
Dn0 = Function(DG).interpolate(vortex_single_elevation_pos(x[0], x[1], H, DeltaH, Lx, Ly))


# Define weak problem:
W = RT * DG * RT  # Mixed space for velocity and depth (nonlinear F_h) CR*CR*DG
U_n = Function(W)
u_n, D_n, F_n = U_n.split()
D_n.assign(Dn0)

U_p = Function(W)
u_p, D_p, F_h = split(U_p)
U_p.assign(U_n)

# define test functions!
p, q, tF = TestFunctions(W)

h_h = 0.5 * (D_p + D_n)
u_h = 0.5 * (u_p + u_n)

uh_eqn = (inner(p, u_p - u_n) * dx - 0.5 * dt * g * div(p) * h_h * dx
          + q * (D_p - D_n) * dx + 0.5 * dt * H * q * div(u_h) * dx
          + inner(tF, F_h - h_h * u_h) * dx
          )

uh_problem = NonlinearVariationalProblem(uh_eqn, U_p)
params = {'mat_type': 'aij',
          'ksp_type': 'preonly',
          'pc_type': 'lu'}
uh_solver = NonlinearVariationalSolver(uh_problem,
                                       solver_parameters=params)
uh_solver.solve()

u_n.rename('Velocity')
D_n.rename('Depth')

name = "nl_sw"
ufile = File("output/" + name + ".pvd")

tmax = 2  # days
t = 0.
dt1 = 1
dumpfreq = 10
dumpcount = dumpfreq


def dump():
    global dumpcount
    dumpcount += 1
    if (dumpcount > dumpfreq):
        ufile.write(u_n, D_n, time=t)
        dumpcount -= dumpfreq


dump()

while t < tmax / Dt - dt1 / 2:
    t += dt1
    print("t= ", t * Dt)
    uh_solver.solve()
    U_n.assign(U_p)
    ufile.write(u_n, D_n, time=t)


from firedrake import *
H = 10
g = Constant(9.80616)

n = 50
Lx = 5000
Ly = 4330
mesh = PeriodicRectangleMesh(n, n,Lx,Ly)

CR = FunctionSpace(mesh, "RT", 1) # BDM 1 or RT 1 
DG = FunctionSpace(mesh, "DG", 0) #choose what kind of function space?

W = CR*DG # Mixed space for velocity and depth
u,h = TrialFunctions(W)
p,q = TestFunctions(W)


u_0 = pi 
h_0 = 10.5    #initial condition
u_n = Function(CR).assign(u_0)
h_n = Function(DG).assign(h_0) #fields at time step n

def vortex_single_elevation_pos(x,y,H,DeltaH,Lx,Ly):
    o   = 0.1
    xc1 = (1./2.-o)*Lx
    yc1 = (1./2.-o)*Ly
    xc2 = (1./2.+o)*Lx
    yc2 = (1./2.+o)*Ly
    sx = 1.5*Lx/20.
    sy = 1.5*Ly/20.
    
    xp1 = Lx*sin(pi*(x-xc1)/Lx)/sx/pi
    yp1 = Ly*sin(pi*(y-yc1)/Ly)/sy/pi
    xp2 = 0*Lx*sin(pi*(x-xc2)/Lx)/sx/pi
    yp2 = 0*Ly*sin(pi*(y-yc2)/Ly)/sy/pi
    
    h = H  + DeltaH*(exp(-1./2.*(xp1**2.+yp1**2.)) + exp(-1./2.*(xp2**2.+yp2**2.)) - 4.*pi*sx*sy/Lx/Ly )
    return h


Lx   = 5000 # in km                           # Zonal length
Ly   = 4330 # in km                           # Meridonal length

H       = 10 # in km
DeltaH  = 0.075
theta   = 25./360.*2.*pi  
f       = 2.*sin(theta)*2.*pi  # in 1/days
g       = 9.81*(24.*3600.)**2./1000. # in  km/days^2




x = SpatialCoordinate(mesh) # define (x[0], x[1])
Dn0 = Function(DG).interpolate(vortex_single_elevation_pos(x[0],x[1],H,DeltaH,Lx,Ly))
h_n.assign(Dn0)

h_h=0.5*(h+h_n)
u_h=0.5*(u+u_n)
dt = 1/50
alpha = Constant(0.5)
uh_eqn = ((inner(p, u-u_n) - alpha*dt*g*div(p)*h_h
 + q*h - q*h_n + alpha*dt*H*q*div(u_h))*dx) # weak form 
 
 


wn1 = Function(W) # mixed func. for both fields (n+1)
un1, hn1 = wn1.split() # Split func. for individual fields
uh_problem = LinearVariationalProblem(lhs(uh_eqn),rhs(uh_eqn), wn1)
params = {'mat_type': 'aij',
 'ksp_type': 'preonly',
 'pc_type': 'lu',
 'pc_factor_mat_solver_type': 'mumps'}
uh_solver = LinearVariationalSolver(uh_problem,
 solver_parameters=params)
uh_solver.solve()


u_out = Function(CR, name="Velocity").assign(u_n)
h_out = Function(DG, name="Depth").assign(h_n)
outfile = File("LSW.pvd")
outfile2 = File("h.pvd")
outfile2.write(h_out)

t=0
t_array=[]
while t<1:
    t+=dt
    t_array.append(t)
    uh_solver.solve()
    u_n.assign(un1)
    h_n.assign(hn1)
    
    u_out.assign(u_n)
    h_out.assign(h_n)
    outfile.write(u_out,h_out)
    outfile2.write(h_out)
    

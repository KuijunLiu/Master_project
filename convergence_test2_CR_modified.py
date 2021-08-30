from firedrake import *
import matplotlib.pyplot as plt
from split_initializations import *
import numpy as np
import math
# plt.rc('text', usetex = True)
# import latexify 
# from projector import *

H = 10
Dt = 0.0005
dt = Constant(Dt)
errlist = []
uerrors = []
herrors = []
N = range(10, 100, 10)
for n in N:
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
        eqn2 = avg(inner(v2 , n_vec) * inner(un0 , n_vec)) * dS

        proj_problem = LinearVariationalProblem(eqn1, eqn2, u_hat)
        params = {'ksp_type': 'preonly', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}
        proj_solver = LinearVariationalSolver(proj_problem,
                                            solver_parameters=params)
        proj_solver.solve()

        # from IPython import embed; embed()


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
        r_solver.solve()

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

        tmax = 0.5  # days
        t = 0.
        dt1 = 1
        dumpfreq = 10
        dumpcount = dumpfreq


        def dump():
            global dumpcount
            dumpcount += 1
            if (dumpcount > dumpfreq):
                # ufile.write(u, D, q_out, time=t)
                dumpcount -= dumpfreq


        dump()
        ulist = []
        hlist = []
        
        
        norm0 = norm(un0) + norm(Dn0)

        while t < tmax :
            t += 0.05
            print("t= ", t * Dt)  
            proj_solver.solve() 
            r_solver.solve()
            uh_solver0.solve()
            U1.assign(U + dU)
            uh_solver1.solve()
            U2.assign(0.75*U + 0.25*(U1 + dU))
            uh_solver2.solve()
            U.assign((1.0/3.0)*U + (2.0/3.0)*(U2 + dU))
            dump()
            ulist.append(u)
            hlist.append(D)
        u0norm = norm(un0, norm_type="L2")
        h0norm = norm(Dn0, norm_type="L2")
        uerr = errornorm(ulist[-1], un0, norm_type="L2")
        herr = errornorm(hlist[-1], Dn0, norm_type="L2")
        # norm1 = norm(ulist[-1]) + norm(hlist[-1])
        # uerrors.append(math.log(uerr))
        uerrors.append(math.log(uerr/u0norm))
        herrors.append(math.log(herr/h0norm))

plt.figure()
# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8], [60, '', '', 240, '', '', '', 480])
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8], [60, '', '', 240, '', '', '', 480, ''])
plt.plot(uerrors)
# plt.xlabel('$\sqrt{n_{DOF}}$')
plt.xlabel('square root of number of DOF')
plt.ylabel('L2 norm of relative velocity errors')
# plt.grid(True)
plt.show()
       
plt.figure()
# plt.xticks([0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000], [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8], [60, '', '', 240, '', '', '', 480, ''])
plt.plot(herrors)
plt.xlabel('square root of number of DOF')
plt.ylabel('L2 norm of relative depth errors')
# plt.grid(True)
plt.show()
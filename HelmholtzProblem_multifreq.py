# HelmholtzProblem_multifreq.py

import dolfin as dl
import sys, os
import numpy as np

from hippylib.modeling import PDEProblem
from hippylib.algorithms.linSolvers import PETScLUSolver
from hippylib import STATE, PARAMETER, ADJOINT 
import hippylib as hp

class PML:
    def __init__(self, mesh, box, box_pml, A):
        
        t = [None]*4
        
        for i in range(4):
            t[i] = box_pml[i]-box[i]
            if abs(t[i]) < dl.DOLFIN_EPS:
                t[i]= 1.
            
        self.sigma_x = dl.Expression("(x[0]<xL)*A*(x[0]-xL)*(x[0]-xL)/(tL*tL) + (x[0]>xR)*A*(x[0]-xR)*(x[0]-xR)/(tR*tR)",
                            xL=box[0], xR=box[2], A=A, tL=t[0], tR=t[2], degree = 2 )
        
        self.sigma_y = dl.Expression("(x[1]<yB)*A*(x[1]-yB)*(x[1]-yB)/(tB*tB) + (x[1]>yT)*A*(x[1]-yT)*(x[1]-yT)/(tT*tT)",
                            yB=box[1], yT=box[3], A=A, tB=t[1], tT=t[3], degree = 2 )
        
        physical_domain = dl.AutoSubDomain(lambda x, on_boundary: x[0] >= box[0] \
                                                              and x[0] <= box[2] \
                                                              and x[1] >= box[1] \
                                                              and x[1] <= box[3] )
        
        cell_marker = dl.MeshFunction("size_t", mesh, mesh.geometry().dim())
        cell_marker.set_all(0)
        physical_domain.mark(cell_marker, 1)
        self.dx = dl.Measure("dx", subdomain_data=cell_marker)

class SingleSourceHelmholtzProblem(PDEProblem):
    def __init__(self, Vh, sources_loc, wave_number, pml, dirichlet_bcs=None):
        self.Vh = Vh
        self.wave_number = wave_number
        self.PML = pml
        self.dirichlet_bcs = dirichlet_bcs
        
        self.rhs_fwd = self.generate_state()
        
        if type(sources_loc) is dl.Point:
            ps0 = dl.PointSource(self.Vh[STATE].sub(0), sources_loc, 1.)
            ps0.apply(self.rhs_fwd)
        else:
            for source in sources_loc:
                ps0 = dl.PointSource(self.Vh[STATE].sub(0), source, 1.)
                ps0.apply(self.rhs_fwd)

        self.A  = None
        self.At = None
        self.C = None
        self.Wmu = None
        self.Wmm = None
        self.Wuu = None
        
        self.solver = self._createSolver()
        self.solver_fwd_inc = self._createSolver()
        self.solver_adj_inc = self._createSolver()

        self._fwd_operator_key = None
        self._adj_operator_key = None
        self.factorization_builds = {"forward": 0, "adjoint": 0}
        self.factorization_reuses = {"forward": 0, "adjoint": 0}
        
        # KKT dictionary
        self.KKT = {}
        self._warned_blocks = set()
        print(f"    PDE[{self.freq_key}Hz]: sources={self.n_sources}, "
              f"state={self.Vh[STATE].dim()}, parameter={self.Vh[PARAMETER].dim()}, "
              f"RHS={self.rhs_fwd.size()}")
        
    def varf_handler(self,u,m,p):
        
        k = self.wave_number*dl.exp(m)
        ksquared = k**2
        
        sigma_x = self.PML.sigma_x
        sigma_y = self.PML.sigma_y
        
        Kr = ksquared - sigma_x*sigma_y
        Ki = -k*(sigma_x + sigma_y)
        
        Dr_xx = (ksquared+sigma_x*sigma_y)/(ksquared + sigma_x*sigma_x)
        Dr_yy = (ksquared+sigma_x*sigma_y)/(ksquared + sigma_y*sigma_y)
        Di_xx = k*(sigma_x - sigma_y)/(ksquared + sigma_x*sigma_x)
        Di_yy = k*(sigma_y - sigma_x)/(ksquared + sigma_y*sigma_y)
        
        Dr = dl.as_matrix([[Dr_xx, dl.Constant(0.)], [dl.Constant(0.), Dr_yy]])
        Di = dl.as_matrix([[Di_xx, dl.Constant(0.)], [dl.Constant(0.), Di_yy]])
        
        u1, u2 = dl.split(u)
        p1, p2 = dl.split(p)
        
        form_r = dl.inner(dl.grad(u1), dl.grad(p1))*self.PML.dx(1) \
                 - ksquared*u1*p1*self.PML.dx(1)
                 
        form_i = -dl.inner(dl.grad(u2), dl.grad(p2))*self.PML.dx(1) \
                     + ksquared*u2*p2*self.PML.dx(1)

        
        form_pml_r = dl.inner(Dr*dl.grad(u1), dl.grad(p1))*self.PML.dx(0) \
                 + dl.inner(Di*dl.grad(u2), dl.grad(p1))*self.PML.dx(0) \
                 - Kr*u1*p1*self.PML.dx(0) \
                 - Ki*u2*p1*self.PML.dx(0)
       
        form_pml_i = - dl.inner(Dr*dl.grad(u2), dl.grad(p2))*self.PML.dx(0) \
               + dl.inner(Di*dl.grad(u1), dl.grad(p2))*self.PML.dx(0) \
               + Kr*u2*p2*self.PML.dx(0) \
               - Ki*u1*p2*self.PML.dx(0)
       
        return form_r + form_i + form_pml_r + form_pml_i
        
    def generate_state(self):
        return dl.Function(self.Vh[STATE]).vector()
    
    def generate_parameter(self):
        return dl.Function(self.Vh[PARAMETER]).vector()
    
    def init_parameter(self, m):
        dummy = self.generate_parameter()
        m.init( dummy.mpi_comm(), dummy.local_range() )
    
    def solveFwd(self, state, x):
        u = dl.TrialFunction(self.Vh[STATE])
        m = hp.vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        p = dl.TestFunction(self.Vh[ADJOINT])
        key = tuple(x[PARAMETER].get_local())
        if key != self._fwd_operator_key:
            A = dl.assemble(self.varf_handler(u, m, p))
            if self.dirichlet_bcs:
                for bc in self.dirichlet_bcs:
                    bc.apply(A, self.rhs_fwd)
            self.solver.set_operator(A)
            self._fwd_operator_key = key
            self.factorization_builds["forward"] += 1
        else:
            self.factorization_reuses["forward"] += 1
        self.solver.solve(state, self.rhs_fwd)

        
    def solveAdj(self, adj, x, adj_rhs):
        u = dl.TestFunction(self.Vh[STATE])
        m = hp.vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        p = dl.TrialFunction(self.Vh[ADJOINT])
        key = tuple(x[PARAMETER].get_local())
        if key != self._adj_operator_key:
            Aadj = dl.assemble(self.varf_handler(u, m, p))
            if self.dirichlet_bcs:
                for bc in self.dirichlet_bcs:
                    bc.apply(Aadj)
            self.solver.set_operator(Aadj)
            self._adj_operator_key = key
            self.factorization_builds["adjoint"] += 1
        else:
            self.factorization_reuses["adjoint"] += 1
        if self.dirichlet_bcs:
            for bc in self.dirichlet_bcs:
                bc.apply(adj_rhs)
        self.solver.solve(adj, adj_rhs)
     
    def evalGradientParameter(self, x, out):
        u = hp.vector2Function(x[STATE], self.Vh[STATE])
        m = hp.vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        p = hp.vector2Function(x[ADJOINT], self.Vh[ADJOINT])
        dm = dl.TestFunction(self.Vh[PARAMETER])
        res_form = self.varf_handler(u,m,p)
        out.zero()
        dl.assemble( dl.derivative(res_form, m, dm), tensor=out)
         
    def setLinearizationPoint(self,x, gn_approx):
        u = hp.vector2Function(x[STATE], self.Vh[STATE])
        m = hp.vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        p = hp.vector2Function(x[ADJOINT], self.Vh[ADJOINT])
        x_fun = [u,m,p]
        
        f_form = self.varf_handler(u,m,p)
        
        g_form = [None,None,None]
        for i in range(3):
            g_form[i] = dl.derivative(f_form, x_fun[i])
            
        self.A = dl.assemble(dl.derivative(g_form[ADJOINT],u))
        self.At = dl.assemble(dl.derivative(g_form[STATE],p))
        self.C = dl.assemble(dl.derivative(g_form[ADJOINT],m))
        
        if self.dirichlet_bcs:
            for bc in self.dirichlet_bcs:
                bc.apply(self.A)
                bc.apply(self.At)
        
        self.solver_fwd_inc.set_operator(self.A)
        self.solver_adj_inc.set_operator(self.At)
        
        if gn_approx:
            self.Wmu = None
            self.Wuu = None
            self.Wmm = None
        else:
            self.Wmu = dl.assemble(dl.derivative(g_form[PARAMETER],u))
            self.Wuu = dl.assemble(dl.derivative(g_form[STATE],u))
            self.Wmm = dl.assemble(dl.derivative(g_form[PARAMETER],m))
        
        # Populate KKT
        self.KKT = {}
        self.KKT[(STATE, STATE)] = self.Wuu
        self.KKT[(PARAMETER, STATE)] = self.Wmu
        self.KKT[(PARAMETER, PARAMETER)] = self.Wmm
        self.KKT[(ADJOINT, STATE)] = self.A
        self.KKT[(ADJOINT, PARAMETER)] = self.C
        self.KKT[(STATE, ADJOINT)] = self.At
        self.KKT[(PARAMETER, ADJOINT)] = self.C
                
    def solveIncremental(self, out, rhs, is_adj):
        if is_adj:
            if self.dirichlet_bcs:
                for bc in self.dirichlet_bcs:
                    bc.apply(rhs)
            self.solver_adj_inc.solve(out, rhs)
        else:
            if self.dirichlet_bcs:
                for bc in self.dirichlet_bcs:
                    bc.apply(rhs)
            self.solver_fwd_inc.solve(out,rhs)
    
    def apply_ij(self,i,j, dir, out):   
        if (i, j) in self.KKT and self.KKT[(i, j)] is not None:
            if (i, j) == (PARAMETER, ADJOINT):
                self.KKT[(ADJOINT, PARAMETER)].transpmult(dir, out)
            else:
                self.KKT[(i, j)].mult(dir, out)
        elif (j, i) in self.KKT and self.KKT[(j, i)] is not None:
            self.KKT[(j, i)].transpmult(dir, out)
        else:
            out.zero()
    
    def _createSolver(self):
        """Create solver - use LU for stability (works for single-frequency)."""
        # Direct LU solver is reliable for Helmholtz
        return PETScLUSolver(self.Vh[STATE].mesh().mpi_comm())
            
    def _createLUSolver(self):
        return PETScLUSolver(self.Vh[STATE].mesh().mpi_comm())

class MultiSourceHelmholtzProblem(PDEProblem):
    def __init__(self, Vh, sources_loc, wave_number, pml, dirichlet_bcs=None, verbose = False):
        """
        Initialize multi-source Helmholtz problem (single frequency).
        """
        self.Vh = Vh
        self.verbose = verbose
        self.PML = pml
        self.dirichlet_bcs = dirichlet_bcs
        self.sources_loc = [sources_loc] if isinstance(sources_loc, dl.Point) else list(sources_loc)
        self.n_sources = len(self.sources_loc)

        # Handle wave_number input - single frequency
        if isinstance(wave_number, list):
            self.wave_number = wave_number[0]
        else:
            self.wave_number = wave_number

        # Store frequency as cache key
        self.freq_key = float(self.wave_number)

        # Initialize RHS for forward problem
        self.rhs_fwd = self.generate_state()
        self._build_rhs()

        self.A = None
        self.At = None
        self.C = None
        self.Wmu = None
        self.Wmm = None
        self.Wuu = None

        self.solver = self._createSolver()
        self.solver_fwd_inc = self._createSolver()
        self.solver_adj_inc = self._createSolver()

        self._fwd_operator_key = None
        self._adj_operator_key = None
        self._fwd_m_cache = None  # Store previous m for forward
        self._adj_m_cache = None  # Store previous m for adjoint
        self.factorization_builds = {"forward": 0, "adjoint": 0}
        self.factorization_reuses = {"forward": 0, "adjoint": 0}
        
        # KKT dictionary
        self.KKT = {}
        self._warned_blocks = set()

    def _single_source_varf(self, u, m, p):
        """Variational form for a single source component."""
        k = self.wave_number * dl.exp(m)
        ksquared = k**2

        sigma_x = self.PML.sigma_x
        sigma_y = self.PML.sigma_y

        Kr = ksquared - sigma_x*sigma_y
        Ki = -k*(sigma_x + sigma_y)

        Dr_xx = (ksquared+sigma_x*sigma_y)/(ksquared + sigma_x*sigma_x)
        Dr_yy = (ksquared+sigma_x*sigma_y)/(ksquared + sigma_y*sigma_y)
        Di_xx = k*(sigma_x - sigma_y)/(ksquared + sigma_x*sigma_x)
        Di_yy = k*(sigma_y - sigma_x)/(ksquared + sigma_y*sigma_y)

        Dr = dl.as_matrix([[Dr_xx, dl.Constant(0.)], [dl.Constant(0.), Dr_yy]])
        Di = dl.as_matrix([[Di_xx, dl.Constant(0.)], [dl.Constant(0.), Di_yy]])

        u1, u2 = dl.split(u)
        p1, p2 = dl.split(p)

        form_r = dl.inner(dl.grad(u1), dl.grad(p1))*self.PML.dx(1) \
                 - ksquared*u1*p1*self.PML.dx(1)
        form_i = -dl.inner(dl.grad(u2), dl.grad(p2))*self.PML.dx(1) \
                     + ksquared*u2*p2*self.PML.dx(1)

        form_pml_r = dl.inner(Dr*dl.grad(u1), dl.grad(p1))*self.PML.dx(0) \
                 + dl.inner(Di*dl.grad(u2), dl.grad(p1))*self.PML.dx(0) \
                 - Kr*u1*p1*self.PML.dx(0) \
                 - Ki*u2*p1*self.PML.dx(0)

        form_pml_i = - dl.inner(Dr*dl.grad(u2), dl.grad(p2))*self.PML.dx(0) \
               + dl.inner(Di*dl.grad(u1), dl.grad(p2))*self.PML.dx(0) \
               + Kr*u2*p2*self.PML.dx(0) \
               - Ki*u1*p2*self.PML.dx(0)

        return form_r + form_i + form_pml_r + form_pml_i

    def _build_rhs(self):
        """Build RHS vector with separate sources for each component."""
        rhs_fn = dl.Function(self.Vh[STATE])
        
        for i, source in enumerate(self.sources_loc):
            component_space = self.Vh[STATE].sub(i).sub(0)
            ps0 = dl.PointSource(component_space, source, 1.0)
            ps0.apply(rhs_fn.vector())
        
        self.rhs_fwd = rhs_fn.vector()

    def varf_handler(self, u, m, p):
        """Variational form for multi-source problem."""
        u_subs = dl.split(u)
        p_subs = dl.split(p)
        
        total_form = 0
        for i, (ui, pi) in enumerate(zip(u_subs, p_subs)):
            total_form += self._single_source_varf(ui, m, pi)
        
        return total_form

    def generate_state(self):
        return dl.Function(self.Vh[STATE]).vector()

    def generate_parameter(self):
        return dl.Function(self.Vh[PARAMETER]).vector()

    def init_parameter(self, m):
        dummy = self.generate_parameter()
        m.init(dummy.mpi_comm(), dummy.local_range())

    def solveFwd(self, state, x):
        u = dl.TrialFunction(self.Vh[STATE])
        m_vec = x[PARAMETER]
        m = hp.vector2Function(m_vec, self.Vh[PARAMETER])
        p = dl.TestFunction(self.Vh[ADJOINT])
        
        # Check if m has changed significantly
        m_changed = False
        if self._fwd_m_cache is None:
            m_changed = True
        else:
            diff = m_vec - self._fwd_m_cache
            if diff.norm("linf") > 1e-12:
                m_changed = True
        
        # Rebuild if m changed OR first time
        if self._fwd_operator_key is None or m_changed:
            A = dl.assemble(self.varf_handler(u, m, p))
            if self.dirichlet_bcs:
                for bc in self.dirichlet_bcs:
                    bc.apply(A, self.rhs_fwd)
            self.solver.set_operator(A)
            self._fwd_operator_key = self.freq_key
            self._fwd_m_cache = m_vec.copy()
            self.factorization_builds["forward"] += 1
            if self.verbose:
                print(f"    PDE[{self.freq_key} wave number] forward: assembled matrix {A.size(0)}x{A.size(1)}, "
                    f"RHS={self.rhs_fwd.size()}, factorization=BUILD")
                # Uncomment for debugging:
                # print(f"  [BUILD] Forward factorization for freq {self.freq_key}")
        else:
            self.factorization_reuses["forward"] += 1
            # Uncomment for debugging:
            # print(f"  [REUSE] Forward factorization for freq {self.freq_key}")
        
        self.solver.solve(state, self.rhs_fwd)

    def solveAdj(self, adj, x, adj_rhs):
        u = dl.TestFunction(self.Vh[STATE])
        m_vec = x[PARAMETER]
        m = hp.vector2Function(m_vec, self.Vh[PARAMETER])
        p = dl.TrialFunction(self.Vh[ADJOINT])
        
        # Check if m has changed significantly
        m_changed = False
        if self._adj_m_cache is None:
            m_changed = True
        else:
            diff = m_vec - self._adj_m_cache
            if diff.norm("linf") > 1e-12:
                m_changed = True
        
        # Rebuild if m changed OR first time
        if self._adj_operator_key is None or m_changed:
            Aadj = dl.assemble(self.varf_handler(u, m, p))
            if self.dirichlet_bcs:
                for bc in self.dirichlet_bcs:
                    bc.apply(Aadj)
            self.solver.set_operator(Aadj)
            self._adj_operator_key = self.freq_key
            self._adj_m_cache = m_vec.copy()
            self.factorization_builds["adjoint"] += 1
            if self.verbose:
                print(f"    PDE[{self.freq_key}Hz] adjoint: assembled matrix {Aadj.size(0)}x{Aadj.size(1)}, "
                    f"RHS={adj_rhs.size()}, factorization=BUILD")
            # Uncomment for debugging:
            # print(f"  [BUILD] Adjoint factorization for freq {self.freq_key}")
        else:
            self.factorization_reuses["adjoint"] += 1
            # Uncomment for debugging:
            # print(f"  [REUSE] Adjoint factorization for freq {self.freq_key}")
        
        if self.dirichlet_bcs:
            for bc in self.dirichlet_bcs:
                bc.apply(adj_rhs)
        
        self.solver.solve(adj, adj_rhs)

    def evalGradientParameter(self, x, out):
        u = hp.vector2Function(x[STATE], self.Vh[STATE])
        m = hp.vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        p = hp.vector2Function(x[ADJOINT], self.Vh[ADJOINT])
        dm = dl.TestFunction(self.Vh[PARAMETER])
        res_form = self.varf_handler(u, m, p)
        out.zero()
        dl.assemble(dl.derivative(res_form, m, dm), tensor=out)

    def setLinearizationPoint(self, x, gn_approx):
        u = hp.vector2Function(x[STATE], self.Vh[STATE])
        m = hp.vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        p = hp.vector2Function(x[ADJOINT], self.Vh[ADJOINT])
        x_fun = [u, m, p]

        f_form = self.varf_handler(u, m, p)

        g_form = [None, None, None]
        for i in range(3):
            g_form[i] = dl.derivative(f_form, x_fun[i])

        self.A = dl.assemble(dl.derivative(g_form[ADJOINT], u))
        self.At = dl.assemble(dl.derivative(g_form[STATE], p))
        self.C = dl.assemble(dl.derivative(g_form[ADJOINT], m))

        if self.dirichlet_bcs:
            for bc in self.dirichlet_bcs:
                bc.apply(self.A)
                bc.apply(self.At)

        self.solver_fwd_inc.set_operator(self.A)
        self.solver_adj_inc.set_operator(self.At)

        if gn_approx:
            self.Wmu = None
            self.Wuu = None
            self.Wmm = None
        else:
            self.Wmu = dl.assemble(dl.derivative(g_form[PARAMETER], u))
            self.Wuu = dl.assemble(dl.derivative(g_form[STATE], u))
            self.Wmm = dl.assemble(dl.derivative(g_form[PARAMETER], m))

        # Populate KKT
        self.KKT = {}
        self.KKT[(STATE, STATE)] = self.Wuu
        self.KKT[(PARAMETER, STATE)] = self.Wmu
        self.KKT[(PARAMETER, PARAMETER)] = self.Wmm
        self.KKT[(ADJOINT, STATE)] = self.A
        self.KKT[(ADJOINT, PARAMETER)] = self.C
        self.KKT[(STATE, ADJOINT)] = self.At
        self.KKT[(PARAMETER, ADJOINT)] = self.C
        if self.verbose:
            print(f"    PDE[{self.freq_key}Hz] linearization: A={self.A.size(0)}x{self.A.size(1)}, "
                f"At={self.At.size(0)}x{self.At.size(1)}, C={self.C.size(0)}x{self.C.size(1)}, "
                f"GN={gn_approx}")

    def solveIncremental(self, out, rhs, is_adj):
        if is_adj:
            if self.dirichlet_bcs:
                for bc in self.dirichlet_bcs:
                    bc.apply(rhs)
            self.solver_adj_inc.solve(out, rhs)
        else:
            if self.dirichlet_bcs:
                for bc in self.dirichlet_bcs:
                    bc.apply(rhs)
            self.solver_fwd_inc.solve(out, rhs)

    def apply_ij(self, i, j, dir, out):
        if (i, j) in self.KKT and self.KKT[(i, j)] is not None:
            if (i, j) == (PARAMETER, ADJOINT):
                self.KKT[(ADJOINT, PARAMETER)].transpmult(dir, out)
            else:
                self.KKT[(i, j)].mult(dir, out)
        elif (j, i) in self.KKT and self.KKT[(j, i)] is not None:
            self.KKT[(j, i)].transpmult(dir, out)
        else:
            out.zero()

    def _createSolver(self):
        """Create solver - use LU for stability."""
        return PETScLUSolver(self.Vh[STATE].mesh().mpi_comm())

    def _createLUSolver(self):
        return PETScLUSolver(self.Vh[STATE].mesh().mpi_comm())
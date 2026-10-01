# helmholtz_model.py
import dolfin as dl
import math
import numpy as np
import sys, os
import hippylib as hp

from HelmholtzProblem_multifreq import MultiSourceHelmholtzProblem, PML

def helmholtz_settings():
    """Define settings for Helmholtz problem with sources and sensors at top strip"""
    settings = {
        'ndim': 2,
        'nx': 64,
        'ny': 64,
        'box': [0., 0., 3., 3.],
        'box_pml': [-1., -1., 4., 4.],
        'n_sources': 4,
        'source_loc': [(0.3, 3.4), (1.1, 3.4), (1.9, 3.4), (2.7, 3.4)],
        'source_frequencies': [
            [1000, 1400],
            [2000, 2600],
            [800, 1200],
            [1500, 1800]
        ],
        'n_sensors': 32,
        'sensor_radius': 1.0,
        'frequency': 1000,
        'dirichlet': True,
        'pointwise_std': 1.0,
        'correlation_length': 0.2,
        'mean_function_value': 0.0,
        'noise_variance': 1e-3,
        'pml_amplitude': 50,
        'sound_speed': 343.4,
        'density': 1.204,
        'prior_mean': None,
        # TV REGULARIZATION FLAGS
        'use_tv_regularization': False,  # Set to True to use TV instead of Gaussian
        'tv_alpha': 500,   # TV regularization strength
        'tv_beta': 10,  # L2 smoothing for TV
        'tv_epsilon': 1e-2, # Differentiability parameter
        'verbose': False
    }
    return settings

class NoTopPML(PML):
    """PML class with NO absorption on the top boundary"""
    def __init__(self, mesh, box, box_pml, A):
        t = [None]*4
        
        for i in range(4):
            t[i] = box_pml[i]-box[i]
            if abs(t[i]) < dl.DOLFIN_EPS:
                t[i]= 1.
        
        self.sigma_x = dl.Expression("(x[0]<xL)*A*(x[0]-xL)*(x[0]-xL)/(tL*tL) + (x[0]>xR)*A*(x[0]-xR)*(x[0]-xR)/(tR*tR)",
                            xL=box[0], xR=box[2], A=A, tL=t[0], tR=t[2], degree = 2 )
        
        self.sigma_y = dl.Expression("(x[1]<yB)*A*(x[1]-yB)*(x[1]-yB)/(tB*tB)",
                            yB=box[1], A=A, tB=t[1], degree = 2 )
        
        physical_domain = dl.AutoSubDomain(lambda x, on_boundary: x[0] >= box[0] \
                                                              and x[0] <= box[2] \
                                                              and x[1] >= box[1] \
                                                              and x[1] <= box[3] )
        
        cell_marker = dl.MeshFunction("size_t", mesh, mesh.geometry().dim())
        cell_marker.set_all(0)
        physical_domain.mark(cell_marker, 1)
        self.dx = dl.Measure("dx", subdomain_data=cell_marker)

def HelmholtzPrior(Vh_PARAMETER, pointwise_std, correlation_length, mean=None, anis_diff=None, use_tv=False, tv_alpha=0.1, tv_beta=0.001, tv_epsilon=1e-6):
    """Create prior for Helmholtz problem - Gaussian or TV based on flag."""
    
    if use_tv:
        print("  Using TV regularization (edge-preserving)")
        print(f"    alpha={tv_alpha}, beta={tv_beta}, epsilon={tv_epsilon}")
        
        if TV_AVAILABLE:
            # Use hippylib's TV if available
            try:
                from hippylib.modeling.tv import TV
                prior = TV(Vh_PARAMETER, alpha=tv_alpha, beta=tv_beta, epsilon=tv_epsilon)
                if mean is not None:
                    prior.mean = mean
                return prior
            except Exception as e:
                print(f"  Error using hippylib TV: {e}, falling back to custom TV")
                # Fall through to custom implementation
        
        # Custom TV implementation
        prior = TVPrior(Vh_PARAMETER, alpha=tv_alpha, beta=tv_beta, epsilon=tv_epsilon, mean=mean)
        return prior
    else:
        print("  Using Gaussian regularization (BiLaplacian)")
        delta = 1*10 #*600
        gamma = 1/36 * 10 #* 600
        if anis_diff is None:
            theta0 = 1
            theta1 = 1
            alpha = math.pi / 4.
            anis_diff = dl.CompiledExpression(hp.ExpressionModule.AnisTensor2D(), degree=1)
            anis_diff.set(theta0, theta1, alpha)
        if mean is None:
            return hp.BiLaplacianPrior(Vh_PARAMETER, gamma, delta, anis_diff, robin_bc=True)
        else:
            return hp.BiLaplacianPrior(Vh_PARAMETER, gamma, delta, anis_diff, mean=mean, robin_bc=True)

def create_dirichlet_bcs(Vh, n_sources=1):
    """Create Dirichlet boundary conditions on all four boundaries"""
    zero = dl.Constant((0.0, 0.0))
    
    class AllBoundaries(dl.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary
    
    all_boundaries = AllBoundaries()
    bcs = []
    for i in range(n_sources):
        bcs.append(dl.DirichletBC(Vh[hp.STATE].sub(i), zero, all_boundaries))
    return bcs

class MultiFrequencyHelmholtzModel:
    """
    A multi-frequency Helmholtz model that runs separate single-frequency models
    and combines their misfits.
    """
    def __init__(self, settings):
        """Initialize with multiple frequencies per source."""
        self.settings = settings
        self.n_sources = settings['n_sources']
        self.verbose = settings['verbose']

        # TV flag
        self.use_tv = settings.get('use_tv_regularization', False)
        
        # Get frequencies per source
        self.source_frequencies = settings.get('source_frequencies', 
                                               [[settings['frequency']]] * settings['n_sources'])
        
        # Flatten all (source, frequency) pairs
        self.flat_frequencies = []
        self.freq_to_source = []
        self.freq_to_freq_idx = []
        for s_idx, freqs in enumerate(self.source_frequencies):
            for f_idx, freq in enumerate(freqs):
                self.flat_frequencies.append(freq)
                self.freq_to_source.append(s_idx)
                self.freq_to_freq_idx.append(f_idx)
        
        self.n_frequencies = len(self.flat_frequencies)
        print(f"Multi-frequency model with {self.n_frequencies} (source, frequency) pairs")
        for i, (freq, src) in enumerate(zip(self.flat_frequencies, self.freq_to_source)):
            print(f"  Pair {i+1}: Source {src+1}, Freq {freq}Hz")
        
        # Create base model for a single frequency
        self._setup_base_model()
        
        # Create individual models for each frequency
        self._setup_frequency_models()
        
        # Create combined model
        self._setup_combined_model()
    
    def _setup_base_model(self):
        """Set up the base mesh and function spaces."""
        nx = self.settings['nx']
        ny = self.settings['ny']
        box = self.settings['box']
        box_pml = self.settings['box_pml']
        
        self.mesh = dl.RectangleMesh(
            dl.Point(box_pml[0], box_pml[1]), 
            dl.Point(box_pml[2], box_pml[3]), 
            nx, ny, "crossed"
        )
        
        # Function spaces (single frequency version)
        Vh2 = dl.VectorFunctionSpace(self.mesh, 'Lagrange', 2)
        Vh1 = dl.FunctionSpace(self.mesh, 'Lagrange', 1)
        
        n_sources = self.settings['n_sources']
        try:
            Vh2_multi = dl.MixedFunctionSpace([Vh2] * n_sources)
        except AttributeError:
            try:
                from dolfin.function.functionspace import MixedFunctionSpace
                Vh2_multi = MixedFunctionSpace([Vh2] * n_sources)
            except (ImportError, AttributeError):
                element = dl.MixedElement([Vh2.ufl_element()] * n_sources)
                Vh2_multi = dl.FunctionSpace(self.mesh, element)
        
        self.Vh = [Vh2_multi, Vh1, Vh2_multi]
        self.Vh_parameter = Vh1
        
        # Setup sensors
        self._setup_sensors()
        
        print(f"Base mesh: {nx}x{ny}, cells={self.mesh.num_cells()}, vertices={self.mesh.num_vertices()}")
        print(f"Function spaces: state={Vh2_multi.dim()} ({n_sources} sources x 2 components), "
            f"parameter={Vh1.dim()}, adjoint={Vh2_multi.dim()}")

    def _setup_sensors(self):
        """Set up sensors."""
        box = self.settings['box']
        n_sensors = self.settings['n_sensors']
        
        spacing = (box[2] - box[0]) / (n_sensors - 1) if n_sensors > 1 else 1.0
        targets = []
        for i in range(n_sensors):
            x = box[0] + i * spacing
            y = box[3] + .7
            targets.append((x, y))
        
        self.targets = np.array(targets)
        print(f"Number of observation points: {len(self.targets)}")
        return self.targets
    
    def _setup_frequency_models(self):
        """Create a separate model for each frequency."""
        box = self.settings['box']
        box_pml = self.settings['box_pml']
        sound_speed = self.settings['sound_speed']
        density = self.settings['density']
        pml_amplitude = self.settings['pml_amplitude']
        
        source_loc = self.settings['source_loc']
        sources_loc = [dl.Point(*loc) for loc in source_loc]
        
        pml = NoTopPML(self.mesh, box, box_pml, pml_amplitude)
        
        # Get TV settings
        use_tv = self.settings.get('use_tv_regularization', False)
        tv_alpha = self.settings.get('tv_alpha', 0.1)
        tv_beta = self.settings.get('tv_beta', 0.001)
        tv_epsilon = self.settings.get('tv_epsilon', 1e-6)
        
        # Create models for each unique frequency
        self.frequency_models = {}
        self.frequency_models_list = []
        
        for freq in set(self.flat_frequencies):
            omega = 2. * np.pi * freq
            wave_number = dl.Constant(omega / (sound_speed * density))
            
            if self.settings['dirichlet']:
                dirichlet_bcs = create_dirichlet_bcs(self.Vh, n_sources=self.settings['n_sources'])
            else:
                dirichlet_bcs = None
         
            pde = MultiSourceHelmholtzProblem(
                self.Vh, sources_loc, wave_number, pml, dirichlet_bcs=dirichlet_bcs, verbose = self.verbose
            )
            
            # Setup prior for this frequency model
            prior_mean = self.settings.get('prior_mean', None)
            if prior_mean is not None:
                if not isinstance(prior_mean, dl.Function):
                    mean_func = dl.Function(self.Vh_parameter)
                    if hasattr(prior_mean, 'vector'):
                        mean_func.vector().axpy(1.0, prior_mean.vector())
                    else:
                        mean_func.vector().axpy(1.0, prior_mean)
                    mean_function = mean_func
                else:
                    mean_function = prior_mean
            else:
                mean_function_value = self.settings.get('mean_function_value', 0.0)
                mean_function = dl.project(dl.Constant(mean_function_value), self.Vh_parameter)
            
            # Create prior with TV flag
            prior = HelmholtzPrior(
                self.Vh_parameter, 
                self.settings['pointwise_std'],
                self.settings['correlation_length'],
                mean=mean_function.vector(),
                use_tv=use_tv,
                tv_alpha=tv_alpha,
                tv_beta=tv_beta,
                tv_epsilon=tv_epsilon
            )
            
            # Setup observations
            B = hp.assemblePointwiseObservation(self.Vh[hp.STATE], self.targets)
            misfit = hp.DiscreteStateObservation(
                B,
                noise_variance=self.settings['noise_variance']
            )
            
            model = hp.Model(pde, prior, misfit)
            
            self.frequency_models[freq] = {
                'model': model,
                'pde': pde,
                'prior': prior,
                'misfit': misfit,
                'wave_number': wave_number
            }
            self.frequency_models_list.append((freq, model, pde, prior, misfit))
            print(f"  Frequency {freq}Hz: wave number={float(omega / (sound_speed * density)):.6e}, "
                f"state={self.Vh[hp.STATE].dim()}, parameter={self.Vh[hp.PARAMETER].dim()}, "
                f"observations={B.size(0)}")
        
            print(f"Created {len(self.frequency_models)} frequency models "
              f"from {self.n_frequencies} (source, frequency) pairs")

    def _combined_cost(self, x):
        """Compute combined cost: Σ_frequencies Σ_sources Σ_receivers misfit"""
        total_misfit = 0.0
        reg_cost = 0.0
        
        m = x[hp.PARAMETER]
        i = 0
        # For each frequency, solve forward for ALL sources and compute misfit
        for freq, info in self.frequency_models.items():
            # print(i)
            i+=1
            pde = info['pde']
            misfit = info['misfit']
            
            # Create a local state vector (solves for ALL sources at once)
            state = pde.generate_state()
            
            # Solve forward for this frequency (ALL sources)
            x_local = [state, m, None]
            pde.solveFwd(state, x_local)
            
            # Compute misfit for this frequency (already sums over ALL sources and receivers)
            x_temp = [state, m, None]
            misfit_cost = misfit.cost(x_temp)
            total_misfit += misfit_cost
            if self.verbose:
                print(f"  Freq {freq}Hz: misfit_cost = {misfit_cost:.6e}")
        
        # Compute regularization (once, shared across all frequencies)
        reg_cost = self.prior.cost(m)
        total_cost = total_misfit + reg_cost
        
        return [total_cost, reg_cost, total_misfit]

    def _combined_gradient(self, x, mg, misfit_only=False):
        """Compute combined gradient: Σ_frequencies ∇misfit + ∇reg"""
        mg.zero()
        
        m = x[hp.PARAMETER]
        
        # Initialize combined gradient
        combined_grad = dl.Vector()
        self.prior.init_vector(combined_grad, hp.PARAMETER)
        combined_grad.zero()
        i = 0
        # For each frequency, compute gradient and add
        if self.verbose:
            print(f"[gradient] parameter dimension={m.size()}, frequencies={len(self.frequency_models)}, "
                f"include_prior={not misfit_only}")
        for freq, info in self.frequency_models.items():
            if self.verbose:
                print(f"  [gradient] freq={freq}Hz ({i + 1}/{len(self.frequency_models)})")
            i+=1
            pde = info['pde']
            misfit = info['misfit']
            
            # Create a local state vector (solves for ALL sources at once)
            state = pde.generate_state()
            
            # Solve forward for this frequency (ALL sources)
            x_local = [state, m, None]
            pde.solveFwd(state, x_local)
            
            # Compute adjoint RHS
            # adj_rhs = B^T * (B*u - d) / noise_variance
            adj_rhs = dl.Vector()
            misfit.B.init_vector(adj_rhs, 1)
            residual = dl.Vector()
            misfit.B.init_vector(residual, 0)
            misfit.B.mult(state, residual)
            residual.axpy(-1.0, misfit.d)
            misfit.B.transpmult(residual, adj_rhs)
            adj_rhs *= -(1.0 / misfit.noise_variance)
            if self.verbose:
                print(f"    state={state.size()}, residual={residual.size()}, adjoint_rhs={adj_rhs.size()}")
            
            # Solve adjoint for this frequency (ALL sources)
            adj = pde.generate_state()
            x_adj = [state, m, adj]
            pde.solveAdj(adj, x_adj, adj_rhs)
            
            # Compute gradient for this frequency
            temp_grad = dl.Vector()
            self.prior.init_vector(temp_grad, hp.PARAMETER)
            x_temp = [state, m, adj]
            pde.evalGradientParameter(x_temp, temp_grad)
            if self.verbose:
                print(f"    adjoint={adj.size()}, frequency_gradient={temp_grad.size()}")
            
            # Add to combined gradient
            combined_grad.axpy(1.0, temp_grad)
        
        # Add prior gradient
        if not misfit_only:
            prior_grad = dl.Vector()
            self.prior.init_vector(prior_grad, hp.PARAMETER)
            self.prior.grad(m, prior_grad)
            combined_grad.axpy(1.0, prior_grad)
            if self.verbose:
                print(f"  [gradient] prior_gradient={prior_grad.size()}, combined={combined_grad.size()}")
        
        mg.zero()
        mg.axpy(1.0, combined_grad)
        
        # Compute gradient norm
        tmp = dl.Vector()
        self.prior.init_vector(tmp, hp.PARAMETER)
        self.prior.Msolver.solve(tmp, mg)
        grad_norm = np.sqrt(mg.inner(tmp))
        
        return grad_norm

    def _combined_solve_fwd(self, out, x):
        """Solve forward for ALL frequency models."""
        m = x[hp.PARAMETER]
        
        # Initialize output state
        out.zero()
        
        if self.verbose:
            print(f"[forward] parameter={m.size()}, output state={out.size()}, "
                f"solving {len(self.frequency_models)} frequencies")
        
        # For each frequency, solve forward and store in out
        # Note: This assumes all frequencies share the same state space structure
        for freq, info in self.frequency_models.items():
            pde = info['pde']
            
            # Create a local state vector
            state = pde.generate_state()
            
            # Solve forward for this frequency
            x_local = [state, m, None]
            pde.solveFwd(state, x_local)
            if self.verbose:
                print(f"  [forward] freq={freq}Hz local state={state.size()}")
            
            # Add to output (or store separately)
            # For multi-frequency, you might want to store all states
            # For now, we'll use the first frequency as the primary state
            if freq == self.flat_frequencies[0]:
                out.axpy(1.0, state)
            # Optional: Store other states in a dictionary for later use

    def _combined_solve_adj(self, adj, x):
        """Solve adjoint for ALL frequency models."""
        m = x[hp.PARAMETER]
        state = x[hp.STATE]
        
        # Initialize adjoint output
        adj.zero()
        if self.verbose:
            print(f"[adjoint] parameter={m.size()}, state={state.size()}, output adjoint={adj.size()}, "
                f"solving {len(self.frequency_models)} frequencies")
        
        # For each frequency, solve adjoint
        for freq, info in self.frequency_models.items():
            pde = info['pde']
            misfit = info['misfit']
            
            # Compute adjoint RHS for this frequency
            adj_rhs = dl.Vector()
            misfit.B.init_vector(adj_rhs, 1)
            residual = dl.Vector()
            misfit.B.init_vector(residual, 0)
            misfit.B.mult(state, residual)
            residual.axpy(-1.0, misfit.d)
            misfit.B.transpmult(residual, adj_rhs)
            adj_rhs *= -(1.0 / misfit.noise_variance)
            if self.verbose:
                print(f"  [adjoint] freq={freq}Hz residual={residual.size()}, rhs={adj_rhs.size()}")
            
            # Solve adjoint for this frequency
            adj_local = pde.generate_state()
            x_adj = [state, m, adj_local]
            pde.solveAdj(adj_local, x_adj, adj_rhs)
            
            # Add to combined adjoint
            if freq == self.flat_frequencies[0]:
                adj.axpy(1.0, adj_local)
    
    def _setup_combined_model(self):
        """Create a combined model interface."""
        # Use the first model as the base
        first_freq = self.flat_frequencies[0]
        base = self.frequency_models[first_freq]
        
        self.model = base['model']
        self.prior = base['prior']
        self.pde = base['pde']
        self.misfit = base['misfit']
        
        # Store all misfits for combined cost
        self.all_misfits = [info['misfit'] for info in self.frequency_models.values()]
        
        # Override the cost method to combine all frequencies
        self.model.cost = self._combined_cost
        
        # Override gradient to combine all frequencies
        self.model.evalGradientParameter = self._combined_gradient
        
        # Override solve methods to solve ALL frequencies
        self.model.solveFwd = self._combined_solve_fwd
        self.model.solveAdj = self._combined_solve_adj
    
    def generate_vector(self, component="ALL"):
        """Generate a vector for the combined model."""
        if component == "ALL" or component is None:
            return [self.generate_vector(hp.STATE),
                    self.generate_vector(hp.PARAMETER),
                    self.generate_vector(hp.ADJOINT)]
        elif component == hp.STATE:
            # Use the first model's state
            first_freq = self.flat_frequencies[0]
            return self.frequency_models[first_freq]['model'].generate_vector(hp.STATE)
        elif component == hp.PARAMETER:
            # Fix: use init_vector since prior doesn't have generate_vector
            vec = dl.Vector()
            self.prior.init_vector(vec, hp.PARAMETER)
            return vec
        elif component == hp.ADJOINT:
            first_freq = self.flat_frequencies[0]
            return self.frequency_models[first_freq]['model'].generate_vector(hp.ADJOINT)
        return None
    
    def solveFwd(self, out, x):
        """Solve forward for the first frequency model."""
        key = tuple(x[hp.PARAMETER].get_local())
        # print(f"Freq {self.wave_number}: key changed = {key != self._fwd_operator_key}")
        first_freq = self.flat_frequencies[0]
        self.frequency_models[first_freq]['model'].solveFwd(out, x)
    
    def solveAdj(self, out, x):
        """Solve adjoint for the first frequency model."""
        first_freq = self.flat_frequencies[0]
        self.frequency_models[first_freq]['model'].solveAdj(out, x)
    
    def setPointForHessianEvaluations(self, x, gauss_newton_approx=False):
        """Set linearization point for all frequency models."""
        for freq, info in self.frequency_models.items():
            if hasattr(info['pde'], 'setLinearizationPoint'):
                info['pde'].setLinearizationPoint(x, gauss_newton_approx)

    def factorization_summary(self):
        """Return forward/adjoint factorization build and reuse counts."""
        summary = {}
        for freq, info in self.frequency_models.items():
            pde = info['pde']
            summary[freq] = {
                "builds": dict(pde.factorization_builds),
                "reuses": dict(pde.factorization_reuses),
            }
        return summary

def helmholtz_model_multifreq(settings=None):
    """Create and return multi-frequency Helmholtz model."""
    if settings is None:
        settings = helmholtz_settings()
    return MultiFrequencyHelmholtzModel(settings)
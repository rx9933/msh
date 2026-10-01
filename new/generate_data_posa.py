import os, sys

# sys.path.append(os.environ.get('HIPPYLIB_PATH'))
# sys.path.append(os.environ.get('HIPPYFLOW_PATH'))
# sys.path.append('/home/mili/hippylib')
# sys.path.append('/home/mili/hippyflow')
os.environ['OMP_NUM_THREADS'] = '1'

import dolfin as dl
import hippylib as hp
import hippyflow as hf

from map_point_generator import MAPGenerator, ActiveSubspaceComputer
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--bc', type=str, default='pml', help='pml or free_surface')
parser.add_argument('-n_data', '--n_data', type=int, default=1, help="Max number of total data")
parser.add_argument('-j_rank', '--j_rank', type=int, default=400, help="Rank for Hessian approximation in POSA")
parser.add_argument('-base_dir', '--base_dir', type=str, default="posa_data", help="data directory")
parser.add_argument('--seed', type=int, default=1, help='Base random seed for this job')
parser.add_argument('--job_id', type=int, default=0, help='Unique ID for this parallel job')
parser.add_argument('-as_rank_data', '--as_rank_data', type=int, default=300, help="Rank for data subspace reduction")
parser.add_argument('-as_rank_param', '--as_rank_param', type=int, default=1000, help="Rank for parameter subspace reduction")


# if __name__ == "__main__":
args = parser.parse_args()

PROJECT_NAME = args.bc#"free_surface"
sys.path.append(f"../{PROJECT_NAME}_code")
from helmholtz_model import helmholtz_model, helmholtz_settings
base_dir = f"{args.base_dir}_{PROJECT_NAME}"
print("Running with arguments:", args)
settings = helmholtz_settings()
model = helmholtz_model(settings).model

Vh = model.problem.Vh
  
prior = model.prior  

# Add counters to existing MAPGenerator instance
generator = MAPGenerator(model, base_dir)

# Add counter attributes
generator.forward_solve_count = 0
generator.current_iter_fwd = 0
generator.fwd_solves_per_iter = []
generator.backtrack_steps_per_iter = []
generator.newton_iterations = []

# Patch solveFwd
original_solveFwd = model.solveFwd
def counted_solveFwd(state, x):
    generator.forward_solve_count += 1
    generator.current_iter_fwd += 1
    return original_solveFwd(state, x)
model.solveFwd = counted_solveFwd

# Patch the solver creation
original_make_solver = generator.make_inverse_solver
def make_solver_with_counter():
    solver = original_make_solver()
    original_solve = solver.solve
    
    def counted_solve(x_vec):
        generator.current_iter_fwd = 0
        result = original_solve(x_vec)
        generator.fwd_solves_per_iter.append(generator.current_iter_fwd)
        generator.backtrack_steps_per_iter.append(max(0, generator.current_iter_fwd - 1))
        generator.newton_iterations.append(solver.it)
        return result
    
    solver.solve = counted_solve
    return solver

generator.make_inverse_solver = make_solver_with_counter

# Now run sampling
generator.sample_map_point(n_data=1, seed=42)

# Print results
print(f"\nTotal forward solves: {generator.forward_solve_count}")
print(f"Forward solves per iteration: {generator.fwd_solves_per_iter}")
print(f"Backtracking steps: {generator.backtrack_steps_per_iter}")
print(f"Newton iterations: {generator.newton_iterations}")

map_generator = MAPGenerator(model, base_dir, rng=hp.Random(seed=args.seed))




# 1. Generate MAP points
print("\n=== Generating MAP points ===")
map_generator.sample_map_point(args.n_data, args.seed)

# 2. Compute inverse Jacobians (dparameter/ddata)
print("\n=== Computing inverse Jacobians ===")
map_generator.sample_posa_low_rank_jacobian(args.n_data, args.j_rank, args.seed, oversampling=50)
# map_generator.check_generated_jacobian(0, low_rank=True) 

# 3. Compute forward Jacobians (ddata/dparameter)
print("\n=== Computing forward Jacobians ===") 
map_generator.sample_jacobian_at_map(args.n_data,args.seed)

# 4. Compute active subspaces (optional)
# if hasattr(args, 'as_rank_data'):
print("\n=== Computing Active Subspaces ===")
subspace_computer = ActiveSubspaceComputer(model, args.base_dir)
subspace_computer.compute_active_subspaces(
    n_data=args.n_data,
    rank_data=args.as_rank_data,
    rank_para=args.as_rank_param,
    oversampling=getattr(args, 'oversample', 10)
)

print("\n=== All computations completed ===")
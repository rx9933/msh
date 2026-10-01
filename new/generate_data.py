
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import math
import numpy as np
import matplotlib.pyplot as plt
import dolfin as dl
import sys
import time
from datetime import datetime
# sys.path.append('/workspace/arushi/hippylib')
# sys.path.append('/workspace/arushi/hippyflow')
# sys.path.append(os.environ.get('HIPPYLIB_PATH'))
import hippylib as hp
# sys.path.append(os.environ.get('HIPPYFLOW_PATH'))
import hippyflow as hf

import argparse

def log_time(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

parser = argparse.ArgumentParser()
parser.add_argument('--bc', type=str, default='pml', help='pml or free_surface')
parser.add_argument('--n_data', type=int, default=1, help='Number of data samples to generate')
parser.add_argument('--seed', type=int, default=0, help='Base random seed for this job')
parser.add_argument('--job_id', type=int, default=0, help='Unique ID for this parallel job')
args = parser.parse_args()

PROJECT_NAME = args.bc#"free_surface"
sys.path.append(f"../{PROJECT_NAME}_code")

from helmholtz_model import *
# Log job start
log_time(f"Job {args.job_id} started with seed {args.seed}")

################################################################################
# Set up the model
formulation = 'pointwise'
assert formulation.lower() in ['full_state', 'pointwise']

log_time("Creating model...")
start_time = time.time()
settings = helmholtz_settings()
model = helmholtz_model_multifreq(settings).model
log_time(f"Model created in {time.time() - start_time:.2f} seconds")

################################################################################

# Set up unique data directory per job
data_dir = f'data_{PROJECT_NAME}/{formulation}/job_{args.job_id:02d}/'
os.makedirs(data_dir, exist_ok=True)
log_time(f"Data will be saved to: {data_dir}")

# Initialize random generator
job_seed = args.seed + args.job_id
parRandom = hp.Random(job_seed)
log_time(f"Initialized random generator with seed {job_seed}")

##################################################################################
# Generate training data
log_time("Setting up training data generation...")
Vh = model.problem.Vh
mesh = Vh[hp.STATE].mesh()

if formulation.lower() == 'full_state':
    q_trial = dl.TrialFunction(Vh[hp.STATE])
    q_test = dl.TestFunction(Vh[hp.STATE])
    M = dl.PETScMatrix(mesh.mpi_comm())
    dl.assemble(dl.inner(q_trial,q_test) * dl.dx, tensor=M)
    B = hf.StateSpaceIdentityOperator(M, use_mass_matrix=False)
    output_decoder = None
elif formulation.lower() == 'pointwise':
    log_time("Setting up pointwise observation operator...")
    B = model.misfit.B
    q = dl.Vector()
    B.init_vector(q,0)
    dQ = q.get_local().shape[0]
    output_decoder = np.eye(dQ)
    log_time(f"Pointwise observation setup complete (dQ = {dQ})")
else:
    raise ValueError(f"Unknown formulation: {formulation}")

observable = hf.LinearStateObservable(model.problem,B)
prior = model.prior

log_time("Creating DataGenerator...")
dataGenerator = hf.DataGenerator(observable, prior, parRandom=parRandom)
log_time("DataGenerator created")

nsamples = args.n_data
n_samples_pod = 250
pod_rank = 200

log_time(f"Starting data generation for {nsamples} samples...")
start_gen_time = time.time()

if formulation.lower() == 'full_state':
    dataGenerator.two_step_generate(nsamples, n_samples_pod=n_samples_pod, 
                                  derivatives=(1,0), pod_rank=pod_rank, 
                                  data_dir=data_dir)
elif formulation.lower() == 'pointwise':
    log_time("Starting pointwise generation...")
    gen_start = time.time()
    dataGenerator.generate(nsamples, derivatives=(1,0),
                         output_decoder=output_decoder, 
                         data_dir=data_dir, compress=False)
    gen_time = time.time() - gen_start
    log_time(f"Pointwise generation completed in {gen_time:.2f} seconds")
else: 
    raise ValueError(f"Unknown formulation: {formulation}")

total_time = time.time() - start_gen_time
log_time(f"Data generation completed in {total_time:.2f} seconds")
log_time(f"Job {args.job_id} finished successfully")
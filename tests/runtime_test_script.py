#!/usr/bin/python3
import pickle
from pathlib import Path
import argparse
import re

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run JAX likelihood computation')
parser.add_argument('--n_samples', type=int, required=True, help='Number of samples to generate')
parser.add_argument('--batch_size', type=int, required=True, help='Batch size for lax.map')
parser.add_argument('--data_dump_path', type=str, required=True, help='Path to the bilby generation data dump pickle file')

args = parser.parse_args()

print("Importing JAX")
import jax
import jax.numpy as np
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", 'cpu')
jax.config.update("jax_traceback_filtering", 'off')
print("Importing JAX successful")

from jimgw.core.single_event.data import Data, PowerSpectrum
from jimgw.core.single_event.detector import get_H1, get_L1
from jimgw.core.single_event.likelihood import BaseTransientLikelihoodFD, HeterodynedTransientLikelihoodFD
from jimgw.core.single_event.waveform import RippleIMRPhenomPv2

n_samples = args.n_samples
batch_size = args.batch_size
data_dump_path = Path(args.data_dump_path)

event_name = "Unknown"
match = re.search(r'(GW\d+[^/]*)', str(data_dump_path))
if match:
    event_name = match.group(1)
    base_match = re.search(r'(GW\d+)', event_name)
    if base_match:
        event_name = base_match.group(1)

print(f"Running for event: {event_name}")
print(f"Running with: n_samples={n_samples}, batch_size={batch_size}")
print(f"Data dump path: {data_dump_path}")

## We load in the pickle dump from the bilby run
print(f"Loading data from {data_dump_path}")
with open(data_dump_path, 'rb') as pickled_data:
    bilby_gen_data = pickle.load(pickled_data)
    print(bilby_gen_data)

## Define detectors
bilby_ifos = bilby_gen_data.interferometers
_ = bilby_ifos[0].frequency_domain_strain
_ = bilby_ifos[1].frequency_domain_strain

for ifo in bilby_ifos:
    ifo.minimum_frequency = 30.0

sampling_frequency = float(bilby_gen_data.meta_data['command_line_args']['sampling_frequency'])
reference_frequency = float(bilby_gen_data.meta_data['command_line_args']['reference_frequency'])
trigger_time = bilby_gen_data.trigger_time
# Input duration
duration = float(bilby_gen_data.meta_data['command_line_args']['duration'])
post_trigger_duration = float(bilby_gen_data.meta_data['command_line_args']['post_trigger_duration'])
start_time = trigger_time + post_trigger_duration - duration
print(f"{duration = :.6f}")
# Precise duration of the strain data
strain_start_time = bilby_ifos[0].strain_data.start_time
strain_end_time = bilby_ifos[0].strain_data.time_array[-1]
strain_duration = strain_end_time - strain_start_time
strain_post_trigger_duration = strain_end_time - trigger_time
print(f"{strain_duration = :.6f}")

jim_ifos = [get_H1(), get_L1()]
f_mins = np.array([ifo.minimum_frequency for ifo in bilby_ifos])
f_maxs = np.array([ifo.maximum_frequency for ifo in bilby_ifos])
f_min, f_max = np.min(f_mins), np.max(f_maxs)
for i, jim_ifo in enumerate(jim_ifos):

    bilby_ifo = bilby_ifos[i]
    assert bilby_ifo.name == jim_ifo.name, f"{bilby_ifo.name = } != {jim_ifo.name = }"

    freq_mask = bilby_ifo.frequency_mask
    _ = bilby_ifo.frequency_domain_strain
    
    delta_t = bilby_ifo.time_array[1] - bilby_ifo.time_array[0]
    assert delta_t == 1.0 / sampling_frequency, f"{delta_t = } != {1.0 / sampling_frequency = }"

    jim_data = Data.from_fd(
        fd=bilby_ifo.frequency_domain_strain,
        frequencies=bilby_ifo.frequency_array,
        epoch=strain_start_time,
        name=bilby_ifo.name + '_fd_data',
    )
    print(jim_data)

    jim_psd = PowerSpectrum(
        values=bilby_ifo.power_spectral_density_array,
        frequencies=bilby_ifo.frequency_array,
        name=jim_ifo.name + '_psd',
    )

    jim_ifo.frequency_bounds = (bilby_ifo.minimum_frequency, bilby_ifo.maximum_frequency)
    jim_ifo.set_data(jim_data)
    jim_ifo.set_psd(jim_psd)
    

jim_Pv2 = RippleIMRPhenomPv2(f_ref=reference_frequency)

likelihood = BaseTransientLikelihoodFD(
    jim_ifos, waveform=jim_Pv2, 
    f_min=f_min, f_max=f_max,
    trigger_time=trigger_time, 
)

key = jax.random.PRNGKey(42)
subkey = jax.random.split(key,15)

jim_samples = {
    "M_c": jax.random.uniform(subkey[0], (n_samples,), minval=5.0, maxval=80.0),
    "eta": jax.random.uniform(subkey[1], (n_samples,), minval=0.01, maxval=0.25),
    "s1_x": jax.random.uniform(subkey[2], (n_samples,), minval=-0.99, maxval=0.99),
    "s1_y": jax.random.uniform(subkey[3], (n_samples,), minval=-0.99, maxval=0.99),
    "s1_z": jax.random.uniform(subkey[4], (n_samples,), minval=-0.99, maxval=0.99),
    "s2_x": jax.random.uniform(subkey[5], (n_samples,), minval=-0.99, maxval=0.99),
    "s2_y": jax.random.uniform(subkey[6], (n_samples,), minval=-0.99, maxval=0.99),
    "s2_z": jax.random.uniform(subkey[7], (n_samples,), minval=-0.99, maxval=0.99),
    "iota": jax.random.uniform(subkey[8], (n_samples,), minval=0, maxval=np.pi),
    "d_L": jax.random.uniform(subkey[9], (n_samples,), minval=10.0, maxval=1000.0),
    "phase_c": jax.random.uniform(subkey[10], (n_samples,), minval=0, maxval=2 * np.pi),
    "psi": jax.random.uniform(subkey[11], (n_samples,), minval=0, maxval=np.pi),
    "ra": jax.random.uniform(subkey[12], (n_samples,), minval=0, maxval=2 * np.pi),
    "dec": jax.random.uniform(subkey[13], (n_samples,), minval=-np.pi/2, maxval=np.pi/2),
    "t_c": jax.random.uniform(subkey[14], (n_samples,), minval=-0.01, maxval=0.01),
}

import time 
start_1 = time.time()
likelihood_vals_1 = jax.vmap(likelihood.evaluate)(jim_samples, None)
end_1 = time.time()
vmap_time = end_1 - start_1
print(f"[{event_name}] Likelihood computation for {n_samples} samples with vmap (duration={duration:.2f}s) takes {vmap_time:.2f} seconds")

start_2 = time.time()
likelihood_vals_2 = jax.lax.map(lambda params: likelihood.evaluate(params, None), jim_samples, batch_size=batch_size)
end_2 = time.time()
lax_map_time = end_2 - start_2
print(f"[{event_name}] Likelihood computation for {n_samples} samples with lax.map with batch size {batch_size} (duration={duration:.2f}s) takes {lax_map_time:.2f} seconds")
#!/usr/bin/python3
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ["PATH"] = "/home/samson.leong/texlive/2023/bin/x86_64-linux:" + os.environ["PATH"]
import pickle
from pathlib import Path
import numpy as np

print("Importing JAX")
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
print("Importing JAX successful")

from jimgw.single_event.detector import H1, L1
from jimgw.single_event.likelihood import TransientLikelihoodFD
from jimgw.single_event.waveform import RippleIMRPhenomPv2

from bilby.gw.result import CBCResult
from bilby.gw import WaveformGenerator, GravitationalWaveTransient
from bilby.gw.source import lal_binary_black_hole

## We load in the pickle dump from the bilby run
indir = Path('/home/thomas.ng/project/jim_GWTC3/bilby_runs/outdir/GW150914')
with open(indir / 'data/GW150914_data0_1126259462-391_generation_data_dump.pickle', 'rb') as pickled_data:
    bilby_gen_data = pickle.load(pickled_data)

bilby_result = CBCResult.from_hdf5(
    indir / 'result/GW150914_data0_1126259462-391_analysis_H1L1_result.hdf5'
)
bilby_poste = bilby_result.posterior
bilby_prior = bilby_result.priors

## Define detectors
bilby_ifos = bilby_gen_data.interferometers

jim_ifos = [H1, L1]
for i, jim_ifo in enumerate(jim_ifos):

    bilby_ifo = bilby_ifos[i]
    assert bilby_ifo.name == jim_ifo.name, f"{bilby_ifo.name = } != {jim_ifo.name = }"

    freq_mask = bilby_ifo.frequency_mask
    jim_ifo.frequencies = bilby_ifo.frequency_array[freq_mask]
    jim_ifo.data = bilby_ifo.frequency_domain_strain[freq_mask]
    jim_ifo.psd = bilby_ifo.power_spectral_density_array[freq_mask]

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

jim_Pv2 = RippleIMRPhenomPv2(f_ref=reference_frequency)

likelihood_1 = TransientLikelihoodFD(
    jim_ifos, waveform=jim_Pv2, trigger_time=trigger_time, 
    duration=duration, post_trigger_duration=post_trigger_duration
)

likelihood_2 = TransientLikelihoodFD(
    jim_ifos, waveform=jim_Pv2, trigger_time=trigger_time, 
    duration=strain_duration, post_trigger_duration=strain_post_trigger_duration
)

likelihood_kwargs = bilby_result.meta_data['likelihood']
## Reconstruct Bilby waveform generator
bilby_waveform_generator = WaveformGenerator(
    duration=likelihood_kwargs['duration'],
    sampling_frequency=likelihood_kwargs['sampling_frequency'],
    start_time=likelihood_kwargs['sampling_frequency'],
    frequency_domain_source_model=lal_binary_black_hole,
    waveform_arguments=likelihood_kwargs['waveform_arguments']
)

## Reconstruct Bilby likelihood
likelihood_bilby = GravitationalWaveTransient(
    interferometers=bilby_ifos,
    waveform_generator=bilby_waveform_generator,
    time_marginalization=likelihood_kwargs['time_marginalization'],
    phase_marginalization=likelihood_kwargs['phase_marginalization'],
    distance_marginalization=likelihood_kwargs['distance_marginalization'],
    reference_frame=likelihood_kwargs['reference_frame'],
    jitter_time=True,
    priors=bilby_prior.copy(),
)

## Start sampling from the bilby posterior samples
n_samples = 100
samples_bilby = bilby_poste.sample(n_samples, random_state=42)

keys = ["chirp_mass", "symmetric_mass_ratio", "spin_1x", "spin_1y", "spin_1z", "spin_2x", "spin_2y", "spin_2z", "iota", "luminosity_distance", "phase", "psi", "ra", "dec"]
jim_keys = ["M_c", "eta", "s1_x", "s1_y", "s1_z", "s2_x", "s2_y", "s2_z", "iota", "d_L", "phase_c", "psi", "ra", "dec"]
jim_to_bilby_map = dict(zip(jim_keys, keys))

param_list = []
for i in range(n_samples):
    sample = samples_bilby.iloc[i].to_dict()

    likelihood_bilby.parameters = sample.copy()
    # likelihood_bilby.parameters['geocent_time'] = start_time
    logL_bilby = likelihood_bilby.log_likelihood_ratio()
    # print("Bilby Likelihood:", logL_bilby)

    jim_params = {}
    for key in jim_keys:
        jim_params[key] = sample[jim_to_bilby_map[key]]
    jim_params["t_c"] = sample["geocent_time"] - trigger_time
    logL_jim_1 = likelihood_1.evaluate(jim_params, None)
    logL_jim_2 = likelihood_2.evaluate(jim_params, None)
    # print("Jim Likelihood:", logL_jim)

    row = list(jim_params.values())
    row.append(logL_jim_1)
    row.append(logL_jim_2)
    row.append(logL_bilby)
    param_list.append(tuple(row))

diff_logLs = np.array(param_list, dtype=[(key, "f8") for key in jim_keys + ["t_c", "gps", "logL_jim_1", "logL_jim_2", "logL_bilby"]])
np.save("compare_GW150914_likelihoods.npy", diff_logLs)

import matplotlib.pyplot as plt

fig, Axes = plt.subplots(2, 3, figsize=(3.4 * 3.2, 3.4 * 2.1),
                         constrained_layout=True, sharex=True)
axes = Axes.flatten()

logL_diff_1 = diff_logLs['logL_bilby'] - diff_logLs['logL_jim_1']
logL_diff_2 = diff_logLs['logL_bilby'] - diff_logLs['logL_jim_2']
logL_diff_3 = diff_logLs['logL_bilby'] - samples_bilby['log_likelihood']
logL_diff_4 = diff_logLs['logL_bilby'] - (samples_bilby['H1_log_likelihood'] + samples_bilby['L1_log_likelihood'])

kappa_sq = samples_bilby['H1_matched_filter_snr'].to_numpy().real * samples_bilby['H1_optimal_snr']
H1_logL = kappa_sq - 0.5 * samples_bilby['H1_optimal_snr'] ** 2
kappa_sq = samples_bilby['L1_matched_filter_snr'].to_numpy().real * samples_bilby['L1_optimal_snr']
L1_logL = kappa_sq - 0.5 * samples_bilby['L1_optimal_snr'] ** 2
HL_logL = H1_logL + L1_logL
logL_diff_5 = diff_logLs['logL_bilby'] - HL_logL
logL_diff_6 = HL_logL - diff_logLs['logL_jim_2'] 

ax = axes[0]
ax.scatter(
    diff_logLs['logL_bilby'], logL_diff_1, s=5, alpha=0.9
)
ax.set_title(f'Use input duration = {duration:.6f} s')
ax.set_ylabel(r'$ \ln {\cal L}_{\rm Bilby} - \ln {\cal L}_{\rm Jim}$')

ax = axes[1]
ax.scatter(
    diff_logLs['logL_bilby'], logL_diff_2, s=5, alpha=0.9
)
ax.set_title(f'Use strain duration = {strain_duration:.6f} s')
ax.set_ylabel(r'$ \ln {\cal L}_{\rm Bilby} - \ln {\cal L}_{\rm Jim}$')

ax = axes[2]
ax.scatter(
    diff_logLs['logL_bilby'], logL_diff_3, s=5, alpha=0.9
)
ax.set_title(f'Compare posterior logL')
ax.set_ylabel(r'$ \ln {\cal L}_{\rm Bilby} - \texttt{log\_likelihood}$')

ax = axes[3]
ax.scatter(
    diff_logLs['logL_bilby'], logL_diff_4, s=5, alpha=0.9
)
ax.set_title(f'Compare posterior (H1 + L1) logL')
ax.set_ylabel(r'$ \ln {\cal L}_{\rm Bilby} - \texttt{HL\_log\_likelihood}$')

ax = axes[4]
ax.scatter(
    diff_logLs['logL_bilby'], logL_diff_5, s=5, alpha=0.9
)
ax.set_title(f'Compare (H1 + L1) logL from SNRs')
ax.set_ylabel(r'$ \ln {\cal L}_{\rm Bilby} - \texttt{ HL\_log\_likelihood}$')

ax = axes[5]
ax.scatter(
    HL_logL, logL_diff_6, s=5, alpha=0.9
)
ax.set_title('Compare (H1 + L1) logL from SNRs' +"\n"+ 'with Jim')
ax.set_ylabel(r'$ \texttt{HL\_log\_likelihood} - \ln {\cal L}_{\rm Jim} $')
ax.set_xlabel(r'$\texttt{HL\_log\_likelihood}$')

for ax in axes[:-1]:
    ax.set_xlabel(r'$\ln {\cal L}_{\rm Bilby}$')

fig.suptitle('GW150914')
fig.savefig('compare_GW150914_time_likelihoods.pdf')
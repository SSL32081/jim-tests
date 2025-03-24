#!/usr/bin/python
import os, sys
os.environ['JAX_PLATFORMS'] = 'cpu'
from pathlib import Path
import pickle
import jax
import numpy as np
from jimgw.single_event.detector import H1, L1, V1, GroundBased2G
from jimgw.single_event.likelihood import TransientLikelihoodFD
from jimgw.single_event.waveform import RippleIMRPhenomPv2
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
from bilby.gw.likelihood import GravitationalWaveTransient
from bilby.gw.waveform_generator import WaveformGenerator
from bilby.gw.result import CBCResult
from bilby.gw.source import lal_binary_black_hole
import pandas as pd

csv = pd.read_csv('event_status.csv')
events = csv['Event'].values

keys = ["chirp_mass", "symmetric_mass_ratio", "spin_1x", "spin_1y", "spin_1z", "spin_2x", "spin_2y", "spin_2z", "iota", "luminosity_distance", "phase", "psi", "ra", "dec"]
jim_keys = ["M_c", "eta", "s1_x", "s1_y", "s1_z", "s2_x", "s2_y", "s2_z", "iota", "d_L", "phase_c", "psi", "ra", "dec"]
jim_to_bilby_map = dict(zip(jim_keys, keys))

indir = Path(sys.argv[1])
print(f"{indir = }")

event_dict = {}
event_max_abs_diff = 0
event_logL_min = np.inf
event_logL_max = 0

for event_id in events:
    pickle_file = list((indir / f"bilby_runs/outdir/{event_id}/data/").glob("*data_dump.pickle"))[0]
    with open(pickle_file, "rb") as f:
        data_dump = pickle.load(f)

    bilby_ifos = data_dump.interferometers
    freq_mask = bilby_ifos[0].frequency_mask
    for ifo in bilby_ifos:
        freq_mask *= ifo.frequency_mask

    f_min = ifo.frequency_array[freq_mask][0]
    ## Needs to reset Bilby ifos too
    for ifo in bilby_ifos:
        ifo.frequency_mask = freq_mask
        ifo.minimum_frequency = f_min

    sampling_frequency = float(data_dump.meta_data['command_line_args']['sampling_frequency'])
    reference_frequency = float(data_dump.meta_data['command_line_args']['reference_frequency'])
    trigger_time = data_dump.trigger_time
    # Input duration
    duration = float(data_dump.meta_data['command_line_args']['duration'])
    post_trigger_duration = float(data_dump.meta_data['command_line_args']['post_trigger_duration'])
    start_time = trigger_time + post_trigger_duration - duration
    print(f"{duration = :.6f}")
    # Precise duration of the strain data
    strain_start_time = bilby_ifos[0].strain_data.start_time
    strain_end_time = bilby_ifos[0].strain_data.time_array[-1]
    strain_duration = strain_end_time - strain_start_time
    strain_post_trigger_duration = strain_end_time - trigger_time
    print(f"{strain_duration = :.6f}")

    ifos_list_str = data_dump.interferometers.meta_data.keys()

    jim_ifos: list[GroundBased2G] = []
    for i, ifo_name in enumerate(ifos_list_str):
        bilby_ifo = bilby_ifos[i]
        assert bilby_ifo.name == ifo_name, f"{bilby_ifo.name = } != {ifo_name = }"

        # Opt for the global frequency mask instead
        # freq_mask = bilby_ifo.frequency_mask

        print("Adding interferometer ", ifo_name)
        eval(f'jim_ifos.append({ifo_name})')

        jim_ifos[i].frequencies = bilby_ifo.frequency_array[freq_mask]
        jim_ifos[i].data = bilby_ifo.frequency_domain_strain[freq_mask]
        jim_ifos[i].psd = bilby_ifo.power_spectral_density_array[freq_mask]

    waveform = RippleIMRPhenomPv2(f_ref=reference_frequency)

    likelihood_1 = TransientLikelihoodFD(
        jim_ifos, waveform=waveform, trigger_time=trigger_time,
        duration=duration, post_trigger_duration=post_trigger_duration
    )

    likelihood_2 = TransientLikelihoodFD(
        jim_ifos, waveform=waveform, trigger_time=trigger_time,
        duration=strain_duration, post_trigger_duration=strain_post_trigger_duration
    )

    bilby_poste = CBCResult.from_hdf5(
        list((indir / f"bilby_runs/outdir/{event_id}/final_result/").glob("*result.hdf5"))[0]
        ).posterior

    bilby_waveform_generator = WaveformGenerator(
        duration=duration, start_time=start_time,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=lal_binary_black_hole,
        waveform_arguments=dict(
            waveform_approximant="IMRPhenomPv2",
            reference_frequency=reference_frequency,
            minimum_frequency=f_min,
        )
    )

    likelihood_bilby = GravitationalWaveTransient(
        interferometers=bilby_ifos,
        waveform_generator=bilby_waveform_generator,
    )

    n_samples = 100
    samples_bilby = bilby_poste.sample(n_samples, random_state=42)

    param_list = []
    for i in range(n_samples):
        sample = samples_bilby.iloc[i].to_dict()

        likelihood_bilby.parameters = sample
        logL_bilby = likelihood_bilby.log_likelihood_ratio()

        jim_params = {}
        for key in jim_keys:
            jim_params[key] = sample[jim_to_bilby_map[key]]
        jim_params["t_c"] = sample["geocent_time"] - trigger_time
        logL_jim_1 = likelihood_1.evaluate(jim_params, None)
        logL_jim_2 = likelihood_2.evaluate(jim_params, None)

        jim_params.pop("gmst", None)
        row = list(jim_params.values())
        row.append(logL_jim_1)
        row.append(logL_jim_2)
        row.append(logL_bilby)
        param_list.append(tuple(row))

    diff_logLs = np.array(param_list, dtype=[(key, "f8") for key in jim_keys + ["t_c", "logL_jim_1", "logL_jim_2", "logL_bilby"]])

    diff_1 = diff_logLs["logL_bilby"] - diff_logLs["logL_jim_1"]
    diff_2 = diff_logLs["logL_bilby"] - diff_logLs["logL_jim_2"]
    max_abs_diff = np.max(np.abs([diff_1, diff_2]))

    logL_min = np.min([diff_logLs[key] for key in ['logL_bilby', 'logL_jim_1', 'logL_jim_2']])
    logL_max = np.max([diff_logLs[key] for key in ['logL_bilby', 'logL_jim_1', 'logL_jim_2']])
    logL_min_max = (logL_min, logL_max)

    # Update global logL min max.
    event_logL_min = min(event_logL_min, logL_min)
    event_logL_max = max(event_logL_max, logL_max)
    event_max_abs_diff = max(event_max_abs_diff, max_abs_diff)
    # Save result to event dict.
    event_dict[event_id] = diff_logLs

    # Plot the likelihood comparison
    fig, axes = plt.subplots(1, 2, figsize=(3.4 * 2.3, 3.6),
                             constrained_layout=True)

    for ax in axes:
        ax.set_xlabel(r'$\ln {\cal L}_{\rm Bilby}$')
        ax.plot(logL_min_max, logL_min_max, color="black", linestyle="--", lw=1., label="1:1")
        ax.set_xlim(*logL_min_max)
        ax.set_ylim(*logL_min_max)
        ax.grid(False)

    ax = axes[0]
    scat = ax.scatter(
        diff_logLs['logL_bilby'], diff_logLs["logL_jim_1"],
        c=diff_1, cmap='RdBu_r', s=5, alpha=0.9,
        vmin=-max_abs_diff, vmax=max_abs_diff
    )
    ax.set_title(f'Use input duration = {duration:.6f} s')
    ax.set_ylabel(r"$\ln{\cal L}_{\rm Jim}$")

    ax = axes[1]
    ax.scatter(
        diff_logLs['logL_bilby'], diff_logLs["logL_jim_2"],
        c=diff_2, cmap='RdBu_r', s=5, alpha=0.9,
        vmin=-max_abs_diff, vmax=max_abs_diff
    )
    ax.set_title(f'Use strain duration = {strain_duration:.6f} s')

    cbar = fig.colorbar(scat, ax=ax)
    cbar.set_label(r'$ \Delta \ln {\cal L} = \ln {\cal L}_{\rm Bilby} - \ln {\cal L}_{\rm Jim}$')

    fig.suptitle(event_id)
    fig.savefig(f'figures/compare_{event_id}_likelihoods.png', dpi=300)
    plt.close(fig)

# Plot the likelihood comparison for all events
fig, axes = plt.subplots(1, 2, figsize=(3.4 * 2.3, 3.6), constrained_layout=True)

event_logL_min_max = (event_logL_min, event_logL_max)
for ax in axes:
    ax.set_xlabel(r'$\ln {\cal L}_{\rm Bilby}$')
    ax.plot(event_logL_min_max, event_logL_min_max, color="black", linestyle="--", lw=1., label="1:1")
    ax.set_xlim(*event_logL_min_max)
    ax.set_ylim(*event_logL_min_max)

scat_kwargs = dict(
        cmap='RdYlBu', s=5, alpha=0.9, marker=".",
        vmin=-event_max_abs_diff, vmax=event_max_abs_diff
)
for event, diff_logLs in event_dict.items():
    ax = axes[0]
    scat = ax.scatter(
        diff_logLs['logL_bilby'], diff_logLs["logL_jim_1"],
        c=diff_logLs['logL_jim_1'] - diff_logLs["logL_bilby"],
        **scat_kwargs
    )
    ax = axes[1]
    ax.scatter(
        diff_logLs['logL_bilby'], diff_logLs["logL_jim_2"],
        c=diff_logLs['logL_jim_2'] - diff_logLs["logL_bilby"],
        **scat_kwargs
    )

axes[0].set_ylabel(r"$\ln{\cal L}_{\rm Jim}$")
axes[0].set_title(f'Use input duration')
axes[1].set_title(f'Use strain duration')

cbar = fig.colorbar(scat, ax=axes[1])
cbar.set_label(r'$ \Delta \ln {\cal L} = \ln {\cal L}_{\rm Bilby} - \ln {\cal L}_{\rm Jim}$')

fig.savefig(f'figures/all_events_likelihoods_comparison.png', dpi=300)

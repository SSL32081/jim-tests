## Import Jim-relatred stuff

import os
os.environ['JAX_PLATFORMS'] = 'cpu'
import pickle
import ast
import numpy as np

print("Importing JAX")
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
print("Importing JAX successful")

from jimgw.jim import Jim
from jimgw.single_event.detector import H1, L1, V1, GroundBased2G
from jimgw.single_event.likelihood import TransientLikelihoodFD, HeterodynedTransientLikelihoodFD
from jimgw.single_event.waveform import RippleIMRPhenomPv2

from gwpy.timeseries import TimeSeries

# Copy from the GW150914 example from Bilby.

# Note you can get trigger times using the gwosc package, e.g.:
# > from gwosc import datasets
# > datasets.event_gps("GW150914")
trigger_time = 1126259462.4
detectors = ["H1", "L1"]
sampling_frequency = 2048
maximum_frequency = sampling_frequency / 2
minimum_frequency = 20
roll_off = 0.4  # Roll off duration of tukey window in seconds, default is 0.4s
duration = 4  # Analysis segment duration
post_trigger_duration = 2  # Time between trigger time and end of segment
end_time = trigger_time + post_trigger_duration
start_time = end_time - duration

psd_duration = 32 * duration
psd_start_time = start_time - psd_duration
psd_end_time = start_time

jim_ifos = [H1, L1]

# Load public data using GWOSC
for ifo in jim_ifos:
    time_series = TimeSeries.fetch_open_data(ifo.name, start_time, end_time)
    strain_start_time = time_series.epoch.value
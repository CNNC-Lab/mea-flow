import os
import sys
module_path = os.path.abspath("/home/neuro/Desktop/Research/_dev/conic-tools/")
sys.path.insert(0, module_path)
module_path = os.path.abspath("/home/neuro/Desktop/Research/_dev/PySpike/")
sys.path.insert(0, module_path)
# sys.path.insert(0, os.getcwd())

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import scipy.stats as stats

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding, SpectralEmbedding, TSNE
import umap

import NMLfunc as nml

from auxiliary import pairwise_distances, plot_embedding, intrinsic_dimensionality, linear_decoder, reconstruction_loss

import itertools
from numpy import matlib as ml

import pickle
import warnings
warnings.filterwarnings("ignore")

from scipy.io import loadmat
from conic_tools.analysis.signals import SpikeList, StateMatrix
from auxiliary import pairwise_distances

from conic_tools.analysis.metrics.states import compute_timescale


# #######################################################################################
# Analysis parameters
# =======================================================================================
condition = 'n1' # 3 conditions (n1, n2, n3)
well_id = 2 # 4-wells in each condition (labelled 1 to 4)
channels = {
    1: np.arange(0, 16),
    2: np.arange(16, 32),
    3: np.arange(32, 48),
    4: np.arange(48, 64)
}
downsampling_factor = 10 # speed up calculations and conserve some memory
dims = 5 # embedding dimensions (maximum)
# axis_lim = 1 # ??0
K_lle = 10 # ??
lamb = 1 # ??


# #######################################################################################
# Load data
# =======================================================================================
def load_data(conditons, wells):
    state_matrices = []
    for condition in conditons:
        data = loadmat('../../data/{}-DIV17-01.mat'.format(condition))
        ids = data['Channels'][0]
        times = data['Times'][0]
        spk_times = [(i, times[idx] * 1000) for idx, i in enumerate(ids)]
        spk_ids = np.unique(ids)

        sl = SpikeList(spk_times, spk_ids)

        for well_id in wells:
            sl_single = sl.id_slice(list(channels[well_id])).time_slice(100000., 200000.)
            try:
                with open("./plots/comparison/taus/{}_{}_single_well_states.pkl".format(condition, well_id), "rb") as f:
                    states = pickle.load(f)
            except:
                states = sl_single.filter_spiketrains(dt=0.1, tau=20.)
                with open("./plots/comparison/taus/{}_{}_single_well_states.pkl".format(condition, well_id), "wb") as f:
                    pickle.dump(states, f)

            states_single = StateMatrix(states, label="well{}-states".format(well_id), state_var="filtered-spikes",
                                        population="{}".format(condition), standardize=True)
            state_matrices.append(states_single)
    return state_matrices

states = load_data(conditons=['n1', 'n2', 'n3'], wells=[4])


time_axis = np.arange(0., 200000.-100000., 0.1)
taus = []
for idx, state in enumerate(states):
    timescales = compute_timescale(state.matrix, time_axis, max_lag=50000, method=0, plot=True, display=True,
                                   save='plots/comparison/taus/autocorr_comp_{}_'.format(idx), verbose=True, n_procs=16)
    taus.append(timescales)

with open('plots/comparison/taus/100s/autocorr_comp.pkl', 'wb') as f:
    pickle.dump(taus, f)

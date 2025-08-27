import os
import sys
module_path = os.path.abspath("/home/neuro/Desktop/Research/_dev/conic-tools/")
sys.path.insert(0, module_path)
module_path = os.path.abspath("/home/neuro/Desktop/Research/_dev/PySpike/")
sys.path.insert(0, module_path)
# sys.path.insert(0, os.getcwd())

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
            if condition == 'n1' and well_id == 3:
                continue
            elif condition == 'n3' and well_id == 1:
                continue
            else:
                sl_single = sl.id_slice(list(channels[well_id])).time_slice(100000., 200000.)
                try:
                    with open("./plots/comparison/{}/{}_single_well_states.pkl".format(condition, well_id), "rb") as f:
                        states = pickle.load(f)
                except:
                    states = sl_single.filter_spiketrains(dt=0.1, tau=20.)
                    with open("./plots/comparison/{}/{}_single_well_states.pkl".format(condition, well_id), "wb") as f:
                        pickle.dump(states, f)

                states_single = StateMatrix(states, label="well{}-states".format(well_id), state_var="filtered-spikes",
                                            population="{}".format(condition), standardize=True)
                state_matrices.append(states_single)
    return state_matrices

states = load_data(conditons=['n1', 'n2', 'n3'], wells=[1, 2, 3, 4])


# concatenate states
all_states = np.concatenate([x.matrix for x in states], axis=1)
idx = np.random.permutation(all_states.shape[1])
all_states = all_states[:, idx]


# list of labels
labels = list(itertools.chain(*[list(ml.repmat(idx, 1, sd.matrix.shape[1])[0]) for idx, sd in enumerate(states)]))
labels = list(np.array(labels)[idx])
labels = labels[::downsampling_factor]



X = all_states[:, ::downsampling_factor].T
# D = pairwise_distances(X, metric='cosine', save=False)

#
# # ============ PCA =====================
emb = 'PCA'
print("Performing {}".format(emb))
pca = PCA(n_components=dims)
x_embd = pca.fit_transform(X)
x_embd = x_embd / np.max(np.abs(x_embd)) # normalise the values

pca_emb = {'pca': pca, 'x_embd': x_embd}
with open('./plots/comparison/all/{}.pkl'.format(emb), 'wb') as f:
    pickle.dump(pca_emb, f)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
plt.set_cmap('viridis')  # circular cmap
fig.suptitle('PCA')
cmap = labels
scat = ax.scatter(x_embd[:, 0], x_embd[:, 1], x_embd[:, 2], c=cmap, alpha=.7)
legend1 = ax.legend(*scat.legend_elements(), loc="lower left", title="Condition")
ax.add_artist(legend1)
plt.savefig('./plots/comparison/all/comp_{}.png'.format(emb))


emb = 'tSNE'
print("Performing {}".format(emb))
SE = TSNE(n_components=3, metric='euclidean', perplexity=90, random_state=42, n_jobs=-1)
x_embd = SE.fit_transform(X)
x_embd = x_embd / np.max(x_embd)
sne_emb = {'sne': SE, 'x_embd': x_embd}
with open('./plots/comparison/all/{}.pkl'.format(emb), 'wb') as f:
    pickle.dump(sne_emb, f)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
plt.set_cmap('viridis')  # circular cmap
fig.suptitle('tSNE')
cmap = labels
scat = ax.scatter(x_embd[:, 0], x_embd[:, 1], x_embd[:, 2], c=cmap, alpha=.7)
legend1 = ax.legend(*scat.legend_elements(), loc="lower left", title="Condition")
ax.add_artist(legend1)
plt.savefig('./plots/comparison/all/comp_{}.pdf'.format(emb))



D = pairwise_distances(all_states[:, :1000].T, metric='cosine', save=False)
emb = 'Isomap-COS'
print("Performing {}".format(emb))
isomap = Isomap(n_components=dims, n_neighbors=40, n_jobs=-1)
x_embd = isomap.fit_transform(D)
x_embd = x_embd / np.max(x_embd)
isomap_emb = {'isomap': isomap, 'x_embd': x_embd, 'distances': D}
with open('./plots/comparison/all/{}.pkl'.format(emb), 'wb') as f:
    pickle.dump(isomap_emb, f)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
plt.set_cmap('viridis')  # circular cmap
fig.suptitle('Isomap')
cmap = labels[:1000]
scat = ax.scatter(x_embd[:, 0], x_embd[:, 1], x_embd[:, 2], c=cmap, alpha=.7)
legend1 = ax.legend(*scat.legend_elements(), loc="lower left", title="Condition")
ax.add_artist(legend1)
plt.savefig('./plots/comparison/all/comp_{}.pdf'.format(emb))

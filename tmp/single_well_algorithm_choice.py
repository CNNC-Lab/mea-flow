import os
import sys
module_path = os.path.abspath("/home/neurobook/Desktop/Research/_dev/conic-tools/")
sys.path.insert(0, module_path)
module_path = os.path.abspath("/home/neurobook/Desktop/Research/_dev/PySpike/")
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
well_id = 3 # 4-wells in each condition (labelled 1 to 4)
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

# ++++++ Plotting parameters +++++++++++
sns.set(style='ticks', font_scale=1)
colb = ['#377eb8', '#ff7f00', '#4daf4a', '#80af7f',  '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
# plt.set_cmap('hsv') # circular cmap
# make svg text editable in illustrator
# plt.rcParams['svg.fonttype'] = 'none'
# plt.rcParams['pdf.fonttype'] = 42
# plt.rcParams['ps.fonttype'] = 42


# #######################################################################################
# Load data
# =======================================================================================
data = loadmat('../../data/{}-DIV17-01.mat'.format(condition))
ids = data['Channels'][0]
times = data['Times'][0]

spk_times = [(i, times[idx]*1000) for idx, i in enumerate(ids)]
spk_ids = np.unique(ids)

sl = SpikeList(spk_times, spk_ids)
sl_single = sl.id_slice(list(channels[well_id])).time_slice(100000., 200000.)

# plot a sample
sl_single.raster_plot(with_rate=True, save='./plots/single-well/{}/raster_well{}.pdf'.format(condition, well_id))

# #######################################################################################
# Filter spike trains, plot and do a quick analysis
# =======================================================================================
try:
    with open("./plots/single-well/{}/{}_single_well_states.pkl".format(condition, well_id), "rb") as f:
        states = pickle.load(f)
except:
    states = sl_single.filter_spiketrains(dt=0.1, tau=20.)
    with open("./plots/single-well/{}/{}_single_well_states.pkl".format(condition, well_id), "wb") as f:
        pickle.dump(states, f)

states_single = StateMatrix(states, label="well{}-states".format(well_id), state_var="filtered-spikes",
                            population="{}".format(condition), standardize=True)
states_single.plot_sample_traces(save='./plots/single-well/{}/{}_sample_traces.pdf'.format(condition, well_id))
states_single.plot_matrix(save='./plots/single-well/{}/{}_sample_states.pdf'.format(condition, well_id))
states_single.plot_trajectory(save='./plots/single-well/{}/{}_trajectory.pdf'.format(condition, well_id))
effective_dim = states_single.effective_dimensionality(plot=True, display=True,
                        save='./plots/single-well/{}/{}_effective_dimensionality.pdf'.format(condition, well_id))
rank = states_single.rank()
# states_single.state_density(display=True,
#                             save='./plots/single-well/{}/{}_state_density.pdf'.format(condition, well_id))
# #######################################################################################
# Pairwise distances
# =======================================================================================
euc_dist_mat = pairwise_distances(states_single.matrix, metric='euclidean', plot=True, display=True,
                   save='./plots/single-well/{}/{}_Euc_dist.pdf'.format(condition, well_id))
cos_dist_mat = pairwise_distances(states_single.matrix, metric='cosine', plot=True, display=True,
                   save='./plots/single-well/{}/{}_cosine_dist.pdf'.format(condition, well_id))

# #######################################################################################
# Manifold geometry (comparisons and evaluation)
# =======================================================================================
effective_dimensionalities = {}
reconstruction_error = {}
reconstruction_accuracy = {}
reconstruction_precision = {}

X = states_single.matrix[:, ::downsampling_factor].T
D = pairwise_distances(X, metric='cosine', save='./plots/single-well/{}/{}-time-structure.pdf'.format(condition,
                                                                                                      well_id))

def analyse_embedding(emb, X, x_embd, time_axis, axis_lim, dims, K_lle, lamb, condition, well_id,
                      save_path="./plots/single-well/"):
    """
    Utility function to analyse all embeddings
    """
    plot_embedding(x_embd, time_axis, axis_lim, title=emb, display=True,
                   save='{}{}/{}_{}.pdf'.format(save_path, condition, well_id, emb))

    Nneigh, radii, p = intrinsic_dimensionality(x_embd, metric='euclidean', fit='std', thr_start=100, thr_fi=5e3,
                                    save='{}{}/{}_{}-dim.pdf'.format(save_path, condition, well_id, emb))
    rmse, r = linear_decoder(x_embd, dims, cv=10, labels=time_axis)
    rec_loss = reconstruction_loss(X, x_embd, dims, K_lle, lamb, cv=10, plot=True, display=True,
                                   save='{}{}/{}_{}-reconstruction_loss.pdf'.format(save_path, condition,
                                                                                                      well_id, emb))
    return p[0], rmse, r, rec_loss

# ============ PCA =====================
emb = 'PCA'
pca = PCA(n_components=dims)
x_embd = pca.fit_transform(X)
x_embd = x_embd / np.max(np.abs(x_embd)) # normalise the values
time_axis = np.arange(x_embd.shape[0], dtype=int)

effective_dimensionalities[emb], reconstruction_error[emb], reconstruction_accuracy[emb], reconstruction_precision[emb] =\
    analyse_embedding(emb, X, x_embd, time_axis, np.max(x_embd), dims, K_lle, lamb, condition, well_id)

# ============ MDS =====================
emb = 'MDS'
x_embd, eig_mds_dff = nml.cmdscale(D)
x_embd = x_embd / np.max(x_embd)

effective_dimensionalities[emb], reconstruction_error[emb], reconstruction_accuracy[emb], reconstruction_precision[emb] =\
    analyse_embedding(emb, X, x_embd, time_axis, np.max(x_embd), dims, K_lle, lamb, condition, well_id)

# ============ Isomap (COS) =====================
emb = 'Isomap-COS'
isomap = Isomap(n_components=dims, n_neighbors=40)
x_embd = isomap.fit_transform(D)
x_embd = x_embd / np.max(x_embd)

effective_dimensionalities[emb], reconstruction_error[emb], reconstruction_accuracy[emb], reconstruction_precision[emb] =\
    analyse_embedding(emb, X, x_embd, time_axis, np.max(x_embd), dims, K_lle, lamb, condition, well_id)

# ============ Isomap (Minkowski) =====================
emb = 'Isomap-mink'
isomap = Isomap(n_components=dims, n_neighbors=40, metric='minkowski')
x_embd = isomap.fit_transform(X)
x_embd = x_embd / np.max(x_embd)

effective_dimensionalities[emb], reconstruction_error[emb], reconstruction_accuracy[emb], reconstruction_precision[emb] =\
    analyse_embedding(emb, X, x_embd, time_axis, np.max(x_embd), dims, K_lle, lamb, condition, well_id)

# ============ LLE =====================
emb = 'LLE'
lle = LocallyLinearEmbedding(n_components=dims, n_neighbors=60, method='modified')
x_embd = lle.fit_transform(D)
x_embd = x_embd / np.max(x_embd)

effective_dimensionalities[emb], reconstruction_error[emb], reconstruction_accuracy[emb], reconstruction_precision[emb] =\
    analyse_embedding(emb, X, x_embd, time_axis, np.max(x_embd), dims, K_lle, lamb, condition, well_id)

# ============ Spectral =====================
emb = 'Spectral'
SE = SpectralEmbedding(n_components=dims, affinity='nearest_neighbors', n_neighbors=60)
x_embd = SE.fit_transform(D)
x_embd = x_embd / np.max(x_embd)

effective_dimensionalities[emb], reconstruction_error[emb], reconstruction_accuracy[emb], reconstruction_precision[emb] =\
    analyse_embedding(emb, X, x_embd, time_axis, np.max(x_embd), dims, K_lle, lamb, condition, well_id)

# ============ t-SNE =====================
emb = 'tSNE'
SE = TSNE(n_components=3, metric='euclidean', perplexity=90, random_state=42)
x_embd = SE.fit_transform(X)
x_embd = x_embd / np.max(x_embd)

effective_dimensionalities[emb], reconstruction_error[emb], reconstruction_accuracy[emb], reconstruction_precision[emb] =\
    analyse_embedding(emb, X, x_embd, time_axis, np.max(x_embd), dims, K_lle, lamb, condition, well_id)

# ============ UMAP =======================
emb = 'UMAP'
SE = umap.UMAP(n_components=dims, metric='cosine', n_neighbors=70, random_state=42)
x_embd = SE.fit_transform(X)
x_embd = nml.centre_scale(x_embd)

effective_dimensionalities[emb], reconstruction_error[emb], reconstruction_accuracy[emb], reconstruction_precision[emb] =\
    analyse_embedding(emb, X, x_embd, time_axis, np.max(x_embd), dims, K_lle, lamb, condition, well_id)

# ########################################################################################
# Comparisons
# ========================================================================================
plt.figure(figsize=(8,6))
LBL = 'real'
lbl = ['PCA', 'MDS', 'Isomap-COS', 'Isomap-mink', 'LLE', 'Spectral', 'tSNE', 'UMAP']
for n, e in enumerate(lbl):
    plt.bar(n, effective_dimensionalities[e], width=.95, color=colb[n], label=e)
sns.despine()
plt.ylim(0, 3)
plt.ylabel('Dimensionality')
plt.xticks(range(len(lbl)), lbl, rotation=45)
plt.savefig('./plots/single-well/{}/{}-dimensionalities_comparison.pdf'.format(condition, well_id))

sns.set(style='ticks', font_scale=2)
plt.figure(figsize=(10,6))
for n,k in enumerate(reconstruction_accuracy):
    sem = stats.sem(reconstruction_error[k],1, nan_policy='omit')
    plt.errorbar(range(1,dims+1), np.nanmean(reconstruction_error[k],1), yerr=sem, label=k, color=colb[n], alpha=1)
plt.legend(frameon=False, loc=1)
plt.xticks(range(1,dims+1))
plt.ylabel('RMSE')
plt.xlabel('Number of dimensions')
sns.despine()
plt.savefig('./plots/single-well/{}/{}-RMSE_comparison.pdf'.format(condition, well_id))

plt.figure(figsize=(10,6))
for n,k in enumerate(reconstruction_accuracy):
    sem = stats.sem(reconstruction_accuracy[k],1, nan_policy='omit')
    plt.errorbar(range(1,dims+1), np.nanmean(reconstruction_accuracy[k],1), yerr=sem, label=k, color=colb[n], alpha=1)
plt.ylim(0,1)
plt.xticks(range(1,dims+1))
plt.ylabel('Decoding performance [$r$]')
plt.xlabel('Number of dimensions')
plt.legend(frameon=False, loc=5)
sns.despine()
plt.savefig('./plots/single-well/{}/{}-CorrErr_comparison.pdf'.format(condition, well_id))

# Reconstuction correlation
keys = list(reconstruction_precision.keys())
plt.figure(figsize=(10,6))
for n,k in enumerate(keys):
    sem = stats.sem(reconstruction_precision[k],1, nan_policy='omit')
    plt.errorbar(range(1,dims+1), np.nanmean(reconstruction_precision[k],1), yerr=sem, label=k, color=colb[n], alpha=1)
# plt.ylim(0,.8)
plt.ylabel('Reconstruction similarity [$r$]')
plt.xlabel('Number of dimensions')
plt.xticks(range(1,dims+1))
sns.despine()
plt.savefig('./plots/single-well/{}/{}-rec_error_comparison.pdf'.format(condition, well_id))

# Total error
plt.figure(figsize=(8,6))
for n,e in enumerate(lbl):
    plt.bar(n, reconstruction_error[e].mean(), width=.95, color=colb[n], label=e)
sns.despine()
plt.ylim(2000, 2600)
plt.ylabel('Mean RMSE')
plt.xticks(range(len(lbl)), lbl, rotation=45)
plt.savefig('./plots/single-well/{}/{}-RMSE_comparison.pdf'.format(condition, well_id))

# Total reconstruction similarity
plt.figure(figsize=(8,6))
for n,e in enumerate(lbl):
    plt.bar(n, reconstruction_precision[e].mean(), width=.95, color=colb[n], label=e)
sns.despine()
plt.ylim(0.5, 0.8)
plt.ylabel('Mean Reconstruction accuracy')
plt.xticks(range(len(lbl)), lbl, rotation=45)
plt.savefig('./plots/single-well/{}/{}-rec_accuracy_comparison.pdf'.format(condition, well_id))

# Total decoding accuracy
plt.figure(figsize=(8,6))
for n,e in enumerate(lbl):
    plt.bar(n, reconstruction_accuracy[e].mean(), width=.95, color=colb[n], label=e)
sns.despine()
# plt.hlines(1, 6.5, -0.5, color='k', linestyle='--')
# plt.ylim(0.5, 0.8)
plt.ylabel('Linear Decoder accuracy')
plt.xticks(range(len(lbl)), lbl, rotation=45)
plt.savefig('./plots/single-well/{}/{}-decoder_accuracy_comparison.pdf'.format(condition, well_id))

# ########################################################################################
# Save results
# ========================================================================================
results = {
    'states': states_single,
    'rank': rank,
    'euc_dist': euc_dist_mat,
    'cos_dist': cos_dist_mat,
    'dimensionality': effective_dimensionalities,
    'reconstruction_accuracy': reconstruction_accuracy,
    'reconstruction_error': reconstruction_error,
    'reconstruction_precision': reconstruction_precision,
}

with open('./plots/single-well/{}/{}_comparisons.pkl'.format(condition, well_id), 'wb') as f:
    pickle.dump(results, f)

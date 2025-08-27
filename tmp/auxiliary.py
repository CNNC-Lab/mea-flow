import time
import numpy as np
from sklearn import decomposition as sk
from matplotlib import pyplot as plt
import seaborn as sns
import pyspike
from conic_tools.logger import info
from conic_tools.visualization.helper import fig_output
logger = info.get_logger(__name__)


def to_pyspike(spike_list):
    """
    Convert the data in the spike_list to the format used by PySpike
    :param spike_list: SpikeList object
    :return: PySpike SpikeTrain object
    """
    bounds = spike_list.time_parameters()
    spike_trains = []
    for n_train in spike_list.id_list:
        sk_train = spike_list.spiketrains[n_train]
        pyspk_sktrain = pyspike.SpikeTrain(spike_times=sk_train.spike_times, edges=bounds)
        spike_trains.append(pyspk_sktrain)
    return spike_trains


def compute_complete_synchrony(spike_list, n_pairs=500, time_bin=1., tau=20., time_resolved=False, display=True, depth=4):
    """
    Apply various metrics of spike train synchrony
    Note: Has dependency on PySpike package.

    :param spike_list: SpikeList object
    :param n_pairs: number of neuronal pairs to consider in the pairwise correlation measures
    :param time_bin: time_bin (for pairwise correlations)
    :param tau: time constant (for the van Rossum distance)
    :param time_resolved: bool - perform time-resolved synchrony analysis (PySpike)
    :param summary_only: bool - retrieve only a summary of the results
    :param complete: bool - use all metrics or only the ccs (due to computation time, memory)
    :param display: bool - display elapsed time message
    :return results: dict
    """
    if display:
        logger.info("\nAnalysing spike synchrony...")
        t_start = time.time()

    spike_trains = to_pyspike(spike_list)
    results = dict()

    if time_resolved:
        results['SPIKE_sync_profile'] = pyspike.spike_sync_profile(spike_trains)
        results['ISI_profile'] = pyspike.isi_profile(spike_trains)
        results['SPIKE_profile'] = pyspike.spike_profile(spike_trains)

    if depth == 1 or depth == 3:
        results['ccs_pearson'] = spike_list.pairwise_pearson_corrcoeff(n_pairs, time_bin=time_bin, all_coef=False)
        # ccs = spike_list.pairwise_cc(n_pairs, time_bin=time_bin)
        # results['ccs'] = (np.mean(ccs), np.var(ccs))

        if depth >= 3:
            # results['d_vp'] = spike_list.distance_victorpurpura(n_pairs, cost=0.5)
            results['d_vr'] = np.mean(spike_list.distance_van_rossum(tau=tau))
            results['ISI_distance'] = pyspike.isi_distance(spike_trains)
            results['SPIKE_distance'] = pyspike.spike_distance(spike_trains)
            results['SPIKE_sync_distance'] = pyspike.spike_sync(spike_trains)
    else:
        results['ccs_pearson'] = spike_list.pairwise_pearson_corrcoeff(n_pairs, time_bin=time_bin, all_coef=True)
        # results['ccs'] = spike_list.pairwise_cc(n_pairs, time_bin=time_bin)

        if depth >= 3:
            # results['d_vp'] = spike_list.distance_victorpurpura(n_pairs, cost=0.5)
            results['d_vr'] = spike_list.distance_van_rossum(tau=tau)
            results['ISI_distance_matrix'] = pyspike.isi_distance_matrix(spike_trains)
            results['SPIKE_distance_matrix'] = pyspike.spike_distance_matrix(spike_trains)
            results['SPIKE_sync_matrix'] = pyspike.spike_sync_matrix(spike_trains)
            results['ISI_distance'] = pyspike.isi_distance(spike_trains)
            results['SPIKE_distance'] = pyspike.spike_distance(spike_trains)
            results['SPIKE_sync_distance'] = pyspike.spike_sync(spike_trains)

    if display:
        logger.info("Elapsed Time: {0} s".format(str(round(time.time() - t_start, 3))))
    return results


def effective_dimensionality(response_matrix, pca_obj=None, label='', plot=False, display=True, save=False):
    """
    Measure the effective dimensionality of population responses. Based on Abbott et al. (2001). Interactions between
    intrinsic and stimulus-evoked activity in recurrent neural networks.

    :param response_matrix: matrix of continuous responses to analyze (NxT)
    :param pca_obj: if pre-computed, otherwise None
    :param label:
    :param plot:
    :param display:
    :param save:
    :return: (float) dimensionality
    """
    if display:
        logger.info("Determining effective dimensionality...")
        t_start = time.time()
    if pca_obj is None:
        n_features, n_samples = np.shape(response_matrix)  # features = neurons
        if n_features > n_samples:
            logger.warning('WARNING - PCA n_features ({}) > n_samples ({}). Effective dimensionality will be computed '
                           'using {} components!'.format(n_features, n_samples, min(n_samples, n_features)))
        pca_obj = sk.PCA(n_components=min(n_features, n_samples))
    if not hasattr(pca_obj, "explained_variance_ratio_"):
        pca_obj.fit(response_matrix.T)  # we need to transpose here as scipy requires n_samples X n_features
    # compute dimensionality
    dimensionality = 1. / np.sum((pca_obj.explained_variance_ratio_ ** 2))
    if display:
        logger.info("Effective dimensionality = {0}".format(str(round(dimensionality, 2))))
        logger.info("Elapsed Time: {0} s".format(str(round(time.time() - t_start, 3))))
    if plot:
        X = pca_obj.fit_transform(response_matrix.T).T
        plot_pca_dimensionality(dimensionality, pca_obj, X, data_label=label, display=display, save=save)
    return dimensionality


def intrinsic_dimensionality(points, nstep=30, metric='euclidean', dist_mat=None, offset_min=10, win_smooth=7,
                             fit='std', thr_start=10, thr_fi=1e5, plot=True, verbose=False, display=True, save=False):
    """
    Obtain the intrinsic dimensionality of a point cloud / neural population activity
    by using exponent (slope on log-log plot) of cumulative neighbours (NN) distribution
    to estimate intrinsic dimensionality
    :param points: NxT point cloud
    :param nstep: nstep temporal distance between points to evaluate
    :param metric: which metric to obtain distance matrix
    :param dist_mat: if 'precomputed', consider `points` as a distance matrix
    :param offset_min: minimum offset
    :param win_smooth: temporal smoothing window size
    :param fit:
    :param thr_start:
    :param thr_fi:
    :param plot:
    :param verbose:
    :return:
    """
    from sklearn.metrics.pairwise import euclidean_distances, cosine_distances

    Nneigh = np.zeros((points.shape[0], nstep))
    if dist_mat == 'precomputed':
        dist_mat = points
    else:  # compute distance matrix
        # Find diameter of point cloud
        points[np.isnan(points)] = 0
        if metric == 'euclidean':
            dist_mat = euclidean_distances(points)
        elif metric == 'cosine':
            dist_mat = cosine_distances(points)
    dist_mat[dist_mat == 0] = np.nan
    minD = np.nanmin(dist_mat)
    maxD = np.nanmax(dist_mat)

    # Define distances to evaluate
    radii = np.logspace(np.log10(minD), np.log10(maxD), nstep)
    # Fing #neigh vs dist
    for n, rad in enumerate(radii):
        Nneigh[:, n] = np.sum((dist_mat < rad), 1)
        if verbose: print(f'{n + 1}/{len(radii)}')

    # find slope of neighbors increase = dimensionality
    sem = np.std(Nneigh, 0)
    mean_ = np.mean(Nneigh, 0)
    sem_ = np.log10(sem[1:])
    x2p = radii[1:]
    y2p = mean_[1:]
    x2p = x2p / np.max(x2p)  # normalise the distance radii

    # find indeces where curve is linear for fit
    if fit == 'std':  # fit line from #thr_start NN until #thre_fi NN
        start = np.argmin(np.abs(mean_ - thr_start))
        fi = np.argmin(np.abs(mean_ - thr_fi))
    elif fit == 'diff':  # find linear part of the curve using 2nd diff
        diff2nd = np.diff(np.diff(smooth(np.log10(y2p), window_len=win_smooth)))
        fi = np.argmin(diff2nd[offset_min:]) + offset_min
        start = np.argmax(diff2nd[:fi])
    elif fit == 'all':  # use all data
        start = 0
        fi = len(mean_) - 1

    # line fit
    x2fit = x2p[start:fi]
    y2fit = y2p[start:fi]
    p = np.polyfit(np.log10(x2fit), np.log10(y2fit), deg=1)

    # plot
    if plot:
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig, ax = plt.subplots()
        ax.plot(x2p, y2p)  # og data
        y_mod = 10 ** p[1] * np.power(x2fit, p[0])  # best fit power law
        ax.plot(x2fit, y_mod, 'r')
        ax.set_xlabel('Distance')
        ax.set_ylabel('# neighbours')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title('The dimensionality/slope is %.2f' % p[0])
        sns.despine()
        fig_output(fig, display, save)

    return Nneigh, radii, p


def plot_pca_dimensionality(result, pca_obj, display=True, save=False):
    fig7 = plt.figure()
    ax72 = fig7.add_subplot(111)
    ax72.plot(pca_obj.explained_variance_ratio_, 'ob')
    ax72.plot(pca_obj.explained_variance_ratio_, '-b')
    ax72.plot(np.ones_like(pca_obj.explained_variance_ratio_) * result,
              np.linspace(0., np.max(pca_obj.explained_variance_ratio_), len(pca_obj.explained_variance_ratio_)),
              '--r', lw=2.5)
    ax72.set_xlabel(r'PC')
    ax72.set_ylabel(r'Variance Explained')
    ax72.set_xlim([0, round(result) * 2])
    ax72.set_ylim([0, np.max(pca_obj.explained_variance_ratio_)])
    fig_output(fig7, display, save)


def pairwise_distances(X, metric='euclidean', plot=True, display=True, save=False):
    """
    Compute the pairwise distances between rows in a matrix (neurons)
    :param X:
    :return:
    """
    from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

    if metric == 'euclidean':
        dist = euclidean_distances(X)
    elif metric == 'cosine':
        dist = cosine_distances(X)
    else:
        raise NotImplementedError("Distance metric {0} is not implemented".format(metric))

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        pltt = ax.imshow(dist, cmap='viridis', interpolation='none')
        cbar = plt.colorbar(pltt)
        cbar.set_label('{} dissimilarity'.format(metric))
        # ax.set_xlabel('Time [samples]')
        # ax.set_ylabel('Time [samples]')
        fig.suptitle("Average pairwise dissimilarity = {}".format(dist.mean()))
        fig_output(fig, display, save)

    return dist


def plot_embedding(x_emb, labels, AXIS_LIM, ds_plt=1, title="", display=True, save=False):
    """
    Plot embeddings
    :param x_emb: pre-computed embedding
    :param ds_plt: downsampling factor for plots
    :param labels: labels for data points
    :param AXIS_LIM: axis limit
    """
    plt.set_cmap('viridis')  # circular cmap
    MIN = -AXIS_LIM
    fig = plt.figure(figsize=(12, 12))
    grid = fig.add_gridspec(ncols=3, nrows=3)
    plt.suptitle('PCA')
    # 3D projection
    ax = fig.add_subplot(grid[1:, :], projection='3d')
    cmap = labels
    scat = ax.scatter(x_emb[:, 0][::ds_plt], x_emb[:, 1][::ds_plt], x_emb[:, 2][::ds_plt], c=cmap, alpha=.7)
    # cbar = plt.colorbar(scat)
    # cbar.set_label('Angular position')
    # ax.set_xlabel('Comp 1'); ax.set_ylabel('Comp 2'); ax.set_zlabel('Comp 3')
    ax.set_xlim([MIN, AXIS_LIM])
    ax.set_ylim([MIN, AXIS_LIM])
    ax.set_zlim([MIN, AXIS_LIM])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # Comp1 vs Comp2
    ax = fig.add_subplot(grid[0, 0])
    plt.scatter(x_emb[:, 0][::ds_plt], x_emb[:, 1][::ds_plt], c=cmap, alpha=.7)
    ax.set_xlabel('Comp 1');
    plt.ylabel('Comp 2')
    ax.set_xticks([]);
    ax.set_yticks([])
    ax.set_xlim([-AXIS_LIM - .1, AXIS_LIM + .1]);
    ax.set_ylim([-AXIS_LIM - .1, AXIS_LIM + .1])
    sns.despine()
    # Comp2 vs Comp3
    ax = fig.add_subplot(grid[0, 1])
    plt.scatter(x_emb[:, 1][::ds_plt], x_emb[:, 2][::ds_plt], c=cmap, alpha=.7)
    ax.set_xlabel('Comp 2');
    plt.ylabel('Comp 3')
    ax.set_xticks([]);
    ax.set_yticks([])
    ax.set_xlim([-AXIS_LIM - .1, AXIS_LIM + .1]);
    ax.set_ylim([-AXIS_LIM - .1, AXIS_LIM + .1])
    sns.despine()
    # Comp1 vs Comp3
    ax = fig.add_subplot(grid[0, 2])
    plt.scatter(x_emb[:, 0][::ds_plt], x_emb[:, 2][::ds_plt], c=cmap, alpha=.7)
    ax.set_xlabel('Comp 1');
    plt.ylabel('Comp 3')
    ax.set_xticks([]);
    ax.set_yticks([])
    ax.set_xlim([-AXIS_LIM - .1, AXIS_LIM + .1]);
    ax.set_ylim([-AXIS_LIM - .1, AXIS_LIM + .1])
    sns.despine()
    fig.suptitle(title)

    fig_output(fig, display, save)


def linear_decoder(x_emb, dims, cv, labels):
    import NMLfunc as nml
    from sklearn.model_selection import KFold

    N = dims
    Y = labels  # phi[::down].flatten() # angular position
    RMSE = np.zeros((N, cv))
    R = np.zeros((N, cv))
    for n in range(N):  # loop over dimensions
        X_ = x_emb[:, :n + 1]  # first n dimension of mds embedding
        kf = KFold(n_splits=cv)
        for c, (train_index, test_index) in enumerate(kf.split(X_)):  # number of cv folds
            # 80% train - 20%  test
            X_train, X_test = X_[train_index], X_[test_index]
            y_train, y_test = Y[train_index], Y[test_index]
            f = nml.OLE(X_train, y_train)  # f is weight  vector for the dimensions/neurons
            y_pred, rmse, r = nml.LinRec(f, X_test, y_test)
            RMSE[n, c] = rmse
            R[n, c] = r

    return RMSE, R


def reconstruction_loss(X, x_emb, dims, K_lle, LAMBDA, cv=10, plot=True, display=True, save=False):
    import NMLfunc as nml
    from sklearn.model_selection import KFold

    Y = x_emb
    radY = np.percentile(Y - np.mean(Y), 95)
    radX = np.percentile(X - np.mean(X), 95)
    Y_sc = radX / radY * (Y - np.mean(Y))

    rec_corr = np.zeros((dims, cv))
    for dim in range(dims):
        # obtain original high-dim activity and its embedding
        Y = Y_sc[::2, :dim + 1]
        X_ = X[::2].copy()
        kf = KFold(n_splits=cv)
        for c, (train_idx, test_idx) in enumerate(kf.split(X_)):  # number of cv folds
            X_rec = nml.new_LLE_pts(Y[train_idx, :].T, X_[train_idx, :].T, K_lle, Y[test_idx, :].T, LAMBDA)
            #         s0 = np.mean((Y[test_idx].sum(1) - X_[test_idx].sum(1))**2)
            #         s1 = np.mean((X_[test_idx] - X_rec)**2)
            #         s2 = np.mean((X_[test_idx] - np.mean(X_[test_idx]))**2)
            #         var_expl_embd[EMBD][dim,c] = 1 - s0/s2
            #         var_expl[EMBD][dim,c] = 1 - s1/s2
            #         rec_err[EMBD][dim,c] = np.mean(np.sqrt((X_[test_idx]-X_rec)**2))
            real = X_[test_idx].flatten()
            dec = X_rec.flatten()
            rec_corr[dim, c] = np.corrcoef(real, dec)[0, 1]
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        cc = ax[0].imshow(X_[:400], cmap='viridis', aspect='auto')
        plt.colorbar(cc)
        ax[0].set_title('Real data')
        cc1 = ax[1].imshow(X_rec[:400], cmap='viridis', aspect='auto')
        plt.colorbar(cc1)
        ax[1].set_title('Reconstructed')
        fig_output(fig, display, save)
    return rec_corr
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from .clustering import calculate_MAC_matrix
from matplotlib.colors import ListedColormap, BoundaryNorm

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy.stats import qmc
from matplotlib.colors import ListedColormap, BoundaryNorm


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy.stats import qmc
from matplotlib.colors import ListedColormap, BoundaryNorm


# ----------------------------------------------------------------------------- helpers

def _setup(model, mode, germ_bounds, delta):
    Q = model.Q
    clf = model.mode_models[mode - 1].classifier
    if germ_bounds is None:
        phys = np.array([d.get_bounds(delta) for d in Q.variables.values()], dtype=float)
        germ_bounds = Q.variable2germ(phys.T).T
    classes = np.asarray(clf.get_classes())
    to_idx = {c: i for i, c in enumerate(classes)}

    def predict_idx(xi):
        if len(classes) == 1:                       # SubdomainClassifier returns zeros
            return np.zeros(len(xi), dtype=int)
        lab = np.asarray(clf.predict(xi)).astype(int)
        return np.vectorize(to_idx.get)(lab)

    return Q, classes, np.asarray(germ_bounds, dtype=float), predict_idx


def _panel_points(xi_fixed, dims, axes_1d):
    """Grid points (germ space) varying `dims`, all other variables at `xi_fixed`."""
    grids = np.meshgrid(*[axes_1d[d] for d in dims[::-1]])[::-1]   # first dim -> columns
    pts = np.tile(xi_fixed, (grids[0].size, 1))
    for d, g in zip(dims, grids):
        pts[:, d] = g.ravel()
    return pts, grids[0].shape


# ----------------------------------------------------------------------------- slice search

def _visibility(idx, n_cls, min_frac):
    """idx: (n_slices, n_points) class indices -> (counts, fracs)."""
    frac = np.stack([(idx == k).mean(axis=1) for k in range(n_cls)], axis=1)
    return (frac >= min_frac).sum(axis=1), frac


def _score(counts, frac, hc, min_frac):
    """Lexicographic score: visible classes, then the smallest visible area
    (bigger = easier to see), then closeness to the median."""
    vis_min = np.where(frac >= min_frac, frac, np.inf).min(axis=1)
    vis_min[~np.isfinite(vis_min)] = 0
    dist = np.linalg.norm(hc, axis=1)
    return counts * 1e6 + np.minimum(vis_min, 0.05) * 1e4 - dist * 1e-2


def find_slices(model, mode, train_params=None, n_global=20000, n_cand=5000, k_neighbors=100,
                n_verify=64, n_design=32, refine_steps=30, n_grid=30, min_frac=0.003, germ_bounds=None, delta=1e-3,
                seed=0, verbose=False):
    """For every panel of the pairplot, find hidden-variable values at which as many
    classes as possible are visible.

    Idea: class k is visible in the panel at hidden values h exactly when h lies in the
    projection of region k onto the hidden variables. So the best slice is where the
    most projections overlap.
      1. Classify one dense Halton sample of the whole space (plus training points),
         once for all panels.
      2. Per panel, estimate the panel composition at n_cand candidate hidden values
         from the labels of their k_neighbors nearest samples in the hidden variables
         (KD-tree, no classifier calls).
      3. Verify the best n_verify candidates (plus a small space-filling design) exactly
         on a coarse grid, then refine the
         winner with a short local search if it falls short of the estimate.

    Ties are broken by the smallest visible area (larger is easier to see) and then by
    closeness to the median.

    Returns
    -------
    slices : dict {(i, j): xi_fixed (germ space)}, i <= j
    counts : dict {(i, j): number of visible classes}
    """
    from scipy.spatial import cKDTree

    Q, classes, gb, predict_idx = _setup(model, mode, germ_bounds, delta)
    m, n_cls = Q.num_variables(), len(classes)
    rng = np.random.default_rng(seed)
    lo, hi = gb[:, 0], gb[:, 1]
    width = hi - lo
    coarse = [np.linspace(*gb[k], n_grid) for k in range(m)]

    pts = qmc.scale(qmc.Halton(d=m, scramble=True, seed=seed).random(n_global), lo, hi)
    if train_params is not None:
        xt = Q.variable2germ(np.asarray(train_params, dtype=float))
        pts = np.vstack([pts, np.clip(xt, lo, hi)])
    lab = predict_idx(pts)
    u = (pts - lo) / width                                         # scaled to [0, 1]
    design = qmc.scale(qmc.Halton(d=m, scramble=True, seed=seed + 1).random(n_design), lo, hi)

    def evaluate(hc, hidden, dims):
        batch = np.vstack([_panel_points(_with(hidden, h, m), dims, coarse)[0] for h in hc])
        idx = predict_idx(batch).reshape(len(hc), -1)
        cnt, frac = _visibility(idx, n_cls, min_frac)
        return cnt, frac, _score(cnt, frac, hc, min_frac)

    slices, counts_out = {}, {}
    panels = [(i, i) for i in range(m)] + list(combinations(range(m), 2))
    for key in panels:
        dims = sorted(set(key))
        hidden = [k for k in range(m) if k not in dims]
        if not hidden:
            slices[key], counts_out[key] = np.zeros(m), None
            continue

        # --- 1) estimate the panel composition around every sample point ------------
        # The k nearest samples in the hidden variables are spread over the plotted
        # variables, so their labels approximate what the panel at that h would show.
        H = u[:, hidden]
        median_u = ((0 - lo) / width)[hidden]
        cand_u = np.vstack([median_u, H[rng.permutation(len(H))[:n_cand]]])
        _, nn = cKDTree(H).query(cand_u, k=k_neighbors)
        nn_lab = lab[nn]
        est_frac = np.stack([(nn_lab == k).mean(axis=1) for k in range(n_cls)], axis=1)
        cand = cand_u * width[hidden] + lo[hidden]                 # back to germ space
        est_cnt = (est_frac > 0).sum(axis=1)
        est_sc = _score(est_cnt, est_frac, cand, 1e-9)

        # --- 2) verify the most promising candidates exactly ------------------------
        order = np.argsort(-est_sc)
        chosen, seen_cells = [0], set()                            # always verify median
        for i in order:                                            # skip near-duplicates
            cell = tuple(np.floor(cand_u[i] / 0.02).astype(int))
            if cell not in seen_cells:
                seen_cells.add(cell)
                chosen.append(i)
            if len(chosen) >= n_verify:
                break
        hc = np.vstack([cand[chosen], design[:, hidden]])
        cnt, frac, sc = evaluate(hc, hidden, dims)

        # --- 3) short local search if the estimate promised more ----------------------
        target = max(est_cnt.max(), cnt.max())
        cur, cur_s, step = hc[np.argmax(sc)], sc.max(), 0.1
        for _ in range(refine_steps if cnt.max() < target else 0):
            prop = np.clip(cur + rng.normal(0, step, (8, len(hidden))) * width[hidden],
                           lo[hidden], hi[hidden])
            c2, f2, s2 = evaluate(prop, hidden, dims)
            hc, cnt, sc = np.vstack([hc, prop]), np.r_[cnt, c2], np.r_[sc, s2]
            b = int(np.argmax(s2))
            if s2[b] > cur_s:
                cur, cur_s = prop[b], s2[b]
            else:
                step *= 0.7
            if cnt.max() >= target or step < 0.005:
                break

        best = int(np.argmax(sc))
        slices[key], counts_out[key] = _with(hidden, hc[best], m), int(cnt[best])
        if verbose:
            print(f"panel {key}: {cnt[best]}/{n_cls} visible (estimated max {est_cnt.max()}), "
                  f"{len(hc)} slices verified")
    return slices, counts_out


def _with(hidden, values, m):
    xi = np.zeros(m)
    xi[hidden] = values
    return xi


# ----------------------------------------------------------------------------- plotting

def plot_classification_regions(model, mode, slices="auto", n_grid=200, fixed=None,
                                germ_bounds=None, delta=1e-3, train_params=None,
                                figsize=None, annotate=True, **search_kwargs):
    """Pairplot of the subdomain classifier's regions for one mode, using 2D slices.

    Parameters
    ----------
    model : Mosaic
        A trained Mosaic model.
    mode : int
        1-based mode index.
    slices : {"auto", "median"} or dict
        "auto": per panel, search hidden-variable values that show as many classes as
        possible (see find_slices). "median": all hidden variables at their median, or
        at `fixed` if given. A dict as returned by find_slices is also accepted.
    fixed : array_like (n_params,), optional
        Physical values for hidden variables when slices="median".
    train_params : ndarray (n_samples, n_params), optional
        Training parameters in physical units to overlay.
    annotate : bool
        Write the hidden-variable values of each panel into the panel.
    **search_kwargs
        Passed to find_slices (n_candidates, n_grid, min_frac, seed).
    """
    Q, classes, gb, predict_idx = _setup(model, mode, germ_bounds, delta)
    names, m, n_cls = Q.variable_names(), Q.num_variables(), len(classes)

    if isinstance(slices, dict):
        panel_xi = slices
    elif slices == "auto":
        panel_xi, _ = find_slices(model, mode, train_params=train_params, germ_bounds=gb,
                                  delta=delta, **search_kwargs)
    elif slices == "median":
        base = np.zeros(m) if fixed is None else Q.variable2germ(
            np.asarray(fixed, dtype=float).reshape(1, -1))[0]
        panel_xi = {k: base for k in [(i, i) for i in range(m)] + list(combinations(range(m), 2))}
    else:
        raise ValueError("slices must be 'auto', 'median' or a dict")

    xi_axes = [np.linspace(*gb[k], n_grid) for k in range(m)]

    def phys_axis(k, xi_fixed):
        tmp = np.tile(xi_fixed, (n_grid, 1))
        tmp[:, k] = xi_axes[k]
        return Q.germ2variable(tmp)[:, k]

    palette = sns.color_palette("Paired", max(n_cls, 2))[:n_cls]
    cmap = ListedColormap(palette)
    norm = BoundaryNorm(np.arange(n_cls + 1) - 0.5, n_cls)

    seen = set()
    fig, axes = plt.subplots(m, m, figsize=figsize or (3 * m, 3 * m), squeeze=False)
    for r in range(m):
        for c in range(m):
            ax = axes[r, c]
            key = (min(r, c), max(r, c))
            xi_fixed = np.asarray(panel_xi[key], dtype=float)
            px_1d = phys_axis(c, xi_fixed)
            if r == c:
                pts, _ = _panel_points(xi_fixed, [c], xi_axes)
                idx = predict_idx(pts)
                ax.pcolormesh(px_1d, [0, 1], idx[None, :-1], cmap=cmap, norm=norm, shading="flat")
                ax.set_yticks([])
            else:
                py_1d = phys_axis(r, xi_fixed)
                pts, shape = _panel_points(xi_fixed, [r, c], xi_axes)   # rows=r, cols=c
                idx = predict_idx(pts).reshape(shape)
                px, py = np.meshgrid(px_1d, py_1d)
                ax.pcolormesh(px, py, idx, cmap=cmap, norm=norm, shading="nearest", alpha=0.85)
                for k in np.unique(idx):
                    ax.contour(px, py, (idx == k).astype(float), levels=[0.5],
                               colors="k", linewidths=0.8)
                if train_params is not None:
                    tp = np.asarray(train_params)
                    ax.scatter(tp[:, c], tp[:, r], s=6, c="k", alpha=0.4)
                ax.set_ylim(py_1d[0], py_1d[-1])
            ax.set_xlim(px_1d[0], px_1d[-1])
            seen.update(np.unique(idx).tolist())

            hidden = [k for k in range(m) if k not in (r, c)]
            if annotate and hidden:
                vals = Q.germ2variable(xi_fixed.reshape(1, -1))[0]
                txt = "\n".join(f"{names[k]}={vals[k]:.3g}" for k in hidden)
                ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", fontsize=7,
                        bbox=dict(boxstyle="round", fc="white", alpha=0.7, lw=0))

            if r == m - 1:
                ax.set_xlabel(names[c])
            else:
                ax.set_xticklabels([])
            if c == 0:
                ax.set_ylabel(names[r])
            if c != 0 or r == c:
                ax.set_yticklabels([])

    handles = [plt.Rectangle((0, 0), 1, 1, color=palette[i]) for i in range(n_cls)]
    fig.legend(handles, [str(c) for c in classes], title="label", loc="center right")
    how = "searched slices" if not (isinstance(slices, str) and slices == "median") else "median slices"
    title = f"Classification regions in mode {mode} ({how})"
    missing = [str(classes[i]) for i in range(n_cls) if i not in seen]
    if missing:
        title += f"\nnot visible in any panel: {', '.join(missing)}"
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 0.9, 0.95))
    return fig, axes



# def plot_classification_regions(model, mode, n_grid=200, fixed=None, germ_bounds=None,
#                                 delta=1e-3, train_params=None, figsize=None):
#     """Pairplot of the subdomain classifier's regions for one mode, using 2D slices.

#     Each off-diagonal panel evaluates the classifier on an n_grid x n_grid grid over the
#     two plotted variables, with all other variables held at `fixed` (default: the median,
#     i.e. germ value 0). Diagonal panels show the 1D slice along each variable.
#     Evaluation happens directly in germ space, where the classifier was trained.

#     Parameters
#     ----------
#     model : Mosaic
#         A trained Mosaic model.
#     mode : int
#         1-based mode index (as in plot_space_division).
#     n_grid : int
#         Grid resolution per axis.
#     fixed : array_like of shape (n_params,), optional
#         Values (in physical units) for the variables not shown on a panel.
#         Defaults to the median of each distribution.
#     germ_bounds : array_like of shape (n_params, 2), optional
#         Plot range in germ space. Defaults to the [delta, 1-delta] quantiles of Q.
#     train_params : ndarray of shape (n_samples, n_params), optional
#         Training parameters (physical units) to overlay as points.
#     """
#     Q = model.Q
#     clf = model.mode_models[mode - 1].classifier
#     names = Q.variable_names()
#     m = Q.num_variables()

#     # --- plot ranges and slice values, all in germ space -------------------------
#     if germ_bounds is None:
#         phys_bounds = np.array([d.get_bounds(delta) for d in Q.variables.values()])
#         germ_bounds = Q.variable2germ(np.asarray(phys_bounds, dtype=float).T).T
#     germ_bounds = np.asarray(germ_bounds, dtype=float)
#     xi_fixed = np.zeros(m) if fixed is None else Q.variable2germ(
#         np.asarray(fixed, dtype=float).reshape(1, -1))[0]

#     xi_axes = [np.linspace(*germ_bounds[k], n_grid) for k in range(m)]
#     # physical coordinates of each axis (the germ map is monotone per variable)
#     phys_axes = []
#     for k in range(m):
#         tmp = np.tile(xi_fixed, (n_grid, 1))
#         tmp[:, k] = xi_axes[k]
#         phys_axes.append(Q.germ2variable(tmp)[:, k])

#     classes = np.asarray(clf.get_classes())
#     n_cls = len(classes)
#     palette = sns.color_palette("Paired", max(n_cls, 2))[:n_cls]
#     cmap = ListedColormap(palette)
#     norm = BoundaryNorm(np.arange(n_cls + 1) - 0.5, n_cls)
#     to_idx = {c: i for i, c in enumerate(classes)}

#     def predict_idx(xi):
#         lab = np.asarray(clf.predict(xi)).astype(int)
#         if n_cls == 1:                       # SubdomainClassifier returns zeros here
#             return np.zeros(len(xi), dtype=int)
#         return np.vectorize(to_idx.get)(lab)

#     seen = set()
#     fig, axes = plt.subplots(m, m, figsize=figsize or (3 * m, 3 * m), squeeze=False)

#     for r in range(m):          # row -> y variable
#         for c in range(m):      # column -> x variable
#             ax = axes[r, c]
#             if r == c:
#                 xi = np.tile(xi_fixed, (n_grid, 1))
#                 xi[:, c] = xi_axes[c]
#                 idx = predict_idx(xi)
#                 seen.update(idx)
#                 ax.pcolormesh(phys_axes[c], [0, 1], idx[None, :-1], cmap=cmap, norm=norm,
#                               shading="flat")
#                 ax.set_yticks([])
#             else:
#                 gx, gy = np.meshgrid(xi_axes[c], xi_axes[r])
#                 xi = np.tile(xi_fixed, (gx.size, 1))
#                 xi[:, c], xi[:, r] = gx.ravel(), gy.ravel()
#                 idx = predict_idx(xi).reshape(gx.shape)
#                 seen.update(np.unique(idx))
#                 px, py = np.meshgrid(phys_axes[c], phys_axes[r])
#                 ax.pcolormesh(px, py, idx, cmap=cmap, norm=norm, shading="nearest", alpha=0.85)
#                 for k in np.unique(idx):     # crisp borders between regions
#                     ax.contour(px, py, (idx == k).astype(float), levels=[0.5],
#                                colors="k", linewidths=0.8)
#                 if train_params is not None:
#                     tp = np.asarray(train_params)
#                     ax.scatter(tp[:, c], tp[:, r], s=6, c="k", alpha=0.5)
#                 ax.set_ylim(phys_axes[r][0], phys_axes[r][-1])
#             ax.set_xlim(phys_axes[c][0], phys_axes[c][-1])
#             if r == m - 1:
#                 ax.set_xlabel(names[c])
#             else:
#                 ax.set_xticklabels([])
#             if c == 0:
#                 ax.set_ylabel(names[r])
#             if c != 0 or r == c:
#                 ax.set_yticklabels([])

#     handles = [plt.Rectangle((0, 0), 1, 1, color=palette[i]) for i in range(n_cls)]
#     fig.legend(handles, [str(c) for c in classes], title="label", loc="center right")
#     missing = [str(classes[i]) for i in range(n_cls) if i not in seen]
#     title = f"Classification regions in mode {mode} (slices through the median)"
#     if missing:
#         title += f"\nnot intersecting any slice: {', '.join(missing)}"
#     fig.suptitle(title)
#     fig.tight_layout(rect=(0, 0, 0.9, 0.95))
#     return fig, axes

def plot_space_division(model, mode, n_samples=10000):
    """ Return a seaborn PairGrid object showing the division of the variables space for a given mode.
    
        Parameters
        ----------
        model : Mosaic
            A Mosaic model object.
                        
        mode : int
            The mode for which the parametric space division is to be visualized.

        n_samples : int
            The number of variable samples to draw from the model for visualization.
            
        Returns
        -------
        g : seaborn PairGrid object
            A seaborn PairGrid object showing the division of the parametric space for the given mode"""

    if model.n_demo_samples is None or model.n_demo_samples != n_samples:
        model.demo_variables = model.sample(n_samples)
        model.demo_labels = model.get_class_labels(model.demo_variables)
        model.n_demo_samples = n_samples
    
    # labels = model.get_class_labels(model.demo_variables)
    n_clusters = len(np.unique(model.demo_labels[:, mode-1]))

    x_data = pd.DataFrame(model.demo_variables, columns=model.Q.variable_names())
    x_data.insert(min(3, x_data.shape[1]), "label", model.demo_labels[:, mode - 1].astype(int))

    palette=sns.color_palette("Paired", n_clusters)
    g = sns.PairGrid(x_data, hue="label", palette=palette)
    g.map_diag(sns.histplot)
    g.map_offdiag(sns.scatterplot)
    g.add_legend()
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle("Variable space division in mode {}".format(mode))

    return g

def plot_reference_eigenvectors(model, mode):
    """ Return a matplotlib figure showing the reference eigenvectors for a given mode.
    
        Parameters
        ----------
        model : Mosaic
            A Mosaic model object.
                        
        mode : int
            The mode for which the reference eigenvectors are to be visualized.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            A matplotlib figure showing the reference eigenvectors for the given mode"""
    
    reference_eigenvectors = model.get_reference_vectors()[mode-1]

    n_clusters = len(reference_eigenvectors)

    palette=sns.color_palette("Paired", n_clusters)

    colors = iter(palette)


    fig, ax = plt.subplots(n_clusters)
    if n_clusters == 1:
        fig.suptitle("Reference eigenvector of mode {}".format(mode))
    else:
        fig.suptitle("Reference eigenvectors of mode {}".format(mode))

    if n_clusters == 1:
        ax.plot(reference_eigenvectors[0, :], color=next(colors))
    else:
        for i in range(n_clusters):
            ax[i].plot(reference_eigenvectors[i, :], color=next(colors), label=i)
            if i != n_clusters-1:
                ax[i].set_xticklabels([])
                ax[i].set_xticks([])
        ax[i].set_xlabel("Nodes")
        fig.legend()
    return fig

def plot_reference_correlation_matrix(model, mode):
    """ Return a matplotlib figure showing the correlation matrix of the reference eigenvectors for a given mode.
    
        Parameters
        ----------
        model : Mosaic
            A Mosaic model object.
                        
        mode : int
            The mode for which the correlation matrix of the reference eigenvectors is to be visualized.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            A matplotlib figure showing the correlation matrix of the reference eigenvectors for the given mode"""
    
    reference_eigenvectors = model.get_reference_vectors()[mode-1]

    n_clusters = len(reference_eigenvectors)

    
    mac_matrix = calculate_MAC_matrix(reference_eigenvectors, reference_eigenvectors)

    fig = plt.figure(figsize=(6,6))
    fig.suptitle("Correlations between the reference eigenvectors of mode {}".format(mode))
    im = plt.imshow(mac_matrix, interpolation='none', vmin=0, vmax=1, aspect='equal')

    ax = plt.gca()

    ax.set_xticks(np.arange(0, n_clusters, 1))
    ax.set_yticks(np.arange(0, n_clusters, 1))

    ax.set_xlabel("Reference eigenvectors")
    ax.set_ylabel("Reference eigenvectors")

    ax.set_xticklabels(np.arange(0, n_clusters, 1))
    ax.set_yticklabels(np.arange(0, n_clusters, 1))

    ax.set_xticks(np.arange(-.5, n_clusters, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n_clusters, 1), minor=True)

    ax.grid(which='minor', color='k', linestyle='-', linewidth=2)
    ax.tick_params(which='minor', bottom=False, left=False)

    for i in range(n_clusters):
        for j in range(n_clusters):
            text = ax.text(j, i, round(mac_matrix[i, j], 2),
                        ha="center", va="center", color="k")
    cax=fig.add_axes([ax.get_position().x1+0.05,ax.get_position().y0,0.03,ax.get_position().height])
    cbar = plt.colorbar(im, cax = cax)
    cbar.set_label("MAC value")
    
    return fig
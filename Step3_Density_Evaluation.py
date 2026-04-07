# =============================================================================
# STEP 3: DENSITY EVALUATION — EW, LW AND WHOLE SAMPLE
# CT Image Analysis — Master's Thesis
# =============================================================================
# Description:
#   This script computes density statistics for the wood sample and its EW/LW
#   components using the masked CT volume from Step 1. The workflow consists
#   of six sequential stages:
#
#     (1) Loading the masked CT volume and a water-phantom reference scan
#     (2) Water-phantom density calibration (scaling CT values to kg/m³)
#     (3) Hessian-based EW/LW segmentation (findEWLW_distance_filtered) and
#         optional geometric LW refinement (segment_latewood) — same approach
#         as Step 2
#     (4) Visual quality check of the segmentation result
#     (5) [Optional] GMM-based density threshold — applied when the spatial
#         segmentation is deemed insufficient after visual inspection.
#         This is typically required for diffuse-porous hardwood species
#         (e.g. beech, oak) where the EW/LW density contrast is low and the
#         Hessian method alone produces geometrically unreliable results.
#         For ring-porous conifers (spruce, pine) the spatial segmentation
#         is used directly.
#     (6) Descriptive density statistics (mean, mode, std, percentiles,
#         IQR, kurtosis) for the whole sample and separately for EW and LW,
#         followed by CSV export.
#
# Environment: Python 3.13.1 | Jupyter Notebook (.ipynb) | VS Code
# Key packages: numpy, scipy, diplib, scikit-image, sklearn, matplotlib,
#               nrrd, pandas
# =============================================================================


# =============================================================================
# 1. IMPORTS
# =============================================================================

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nrrd
import diplib as dip

from scipy.ndimage import distance_transform_edt
from scipy.stats import mode, kurtosis
from sklearn.mixture import GaussianMixture
from matplotlib.colors import ListedColormap, BoundaryNorm
from skimage.measure import label, regionprops


# =============================================================================
# 2. CONFIGURATION
# =============================================================================

# --- File paths ---
MASK_PATH   = r'C:\Users\jansi\OneDrive\Dokumenti\Master_thesis\CT_Data\MY_SAMPLES_Mask\Spruce\S3-50_MASK.nrrd'
WATER_PATH  = r'C:\Users\jansi\OneDrive\Dokumenti\Master_thesis\CT_Data\Samples_Master_thesis\Oak\water.nrrd'
CSV_PATH    = r'C:\Users\jansi\OneDrive\Dokumenti\Master_thesis\CT_Data\Density_DATA\statistika_vzorca.csv'
SAMPLE_ID   = 'S3-50_MASK'

# --- Water calibration ---
WATER_HU_MIN = 900    # Lower HU bound for selecting water voxels
WATER_HU_MAX = 1100   # Upper HU bound for selecting water voxels

# --- Primary segmentation parameters ---
DISTANCE_THRESH  = 3      # Minimum distance (px) from wood boundary
HESSIAN_SIGMAS   = [1]    # Gaussian smoothing scale for Hessian computation

# --- segment_latewood parameters ---
LW_MIN_AREA    = 10   # Minimum connected LW region area (px²)
LW_BORDER_TOL  = 5    # Edge contact tolerance (px)
LW_MIN_RATIO   = 1.5  # Minimum width/height ratio for a valid LW band

# --- Segmentation method selection ---
# After running the spatial segmentation and inspecting the visual output
# (Section 5), set USE_GMM to True if the result is insufficient.
#   False → use spatial labels (findEWLW + segment_latewood) — recommended
#           for ring-porous conifers (spruce, pine)
#   True  → use GMM density threshold — recommended for diffuse-porous
#           hardwoods (beech, oak) where spatial segmentation is unreliable
USE_GMM = False

# --- GMM threshold choice (used only if USE_GMM = True) ---
# 'midpoint'     → arithmetic mean of the two Gaussian means (more robust)
# 'intersection' → algebraic intersection of the two weighted Gaussians
GMM_THRESHOLD_MODE = 'midpoint'

# --- Visualisation ---
SLICE_PREVIEW  = 120               # Default slice index (Z-axis)
CUSTOM_COLORS  = ['#1f77b4', '#a0522d', '#00ced1']   # BG, EW, LW


# =============================================================================
# 3. LOAD CT DATA
# =============================================================================

img_data,  _ = nrrd.read(MASK_PATH)
img_water, _ = nrrd.read(WATER_PATH)
print(f"Masked volume shape : {img_data.shape}")
print(f"Water phantom shape : {img_water.shape}")

# Visual check: masked volume and water phantom side by side
fig, axs = plt.subplots(1, 2, figsize=(14, 7))
axs[0].imshow(img_data [:, :, SLICE_PREVIEW], cmap='gray', origin='lower')
axs[0].set_title(f"Masked CT Volume (Z = {SLICE_PREVIEW})")
axs[0].axis('off')
axs[1].imshow(img_water[:, :, SLICE_PREVIEW], cmap='gray', origin='lower')
axs[1].set_title(f"Water Phantom (Z = {SLICE_PREVIEW})")
axs[1].axis('off')
plt.tight_layout()
plt.show()


# =============================================================================
# 4. WATER-PHANTOM DENSITY CALIBRATION
# =============================================================================
# The CT scanner outputs arbitrary HU values that must be scaled so that the
# density of water corresponds to 1000 kg/m³.  A water-phantom scan acquired
# under identical scanner settings is used to derive the scale factor.

water_voxels   = img_water[(img_water > WATER_HU_MIN) & (img_water < WATER_HU_MAX)]
mean_water_ct  = np.mean(water_voxels)
scale_factor   = 1000.0 / mean_water_ct
img_data_cal   = img_data * scale_factor   # calibrated volume [kg/m³]

print(f"Water phantom mean CT value : {mean_water_ct:.2f} HU")
print(f"Calibration scale factor    : {scale_factor:.4f}")
print(f"Mean density before calib.  : {np.mean(img_data[img_data > 0]):.2f} HU")
print(f"Mean density after  calib.  : {np.mean(img_data_cal[img_data_cal > 0]):.2f} kg/m³")

# Whole-sample histogram (calibrated, background excluded)
values_all = img_data_cal[img_data_cal > 0].ravel()

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(values_all, bins=200, color='gray', edgecolor='black')
ax.set_title(f"Whole-Sample Density Histogram  (n = {values_all.size:,})")
ax.set_xlabel("Density [kg/m³]")
ax.set_ylabel("Number of Voxels")
ax.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()


# =============================================================================
# 5. SPATIAL EW/LW SEGMENTATION
# =============================================================================
# The same two-stage approach used in Step 2 is applied here on the calibrated
# volume.  First, the Hessian curvature method separates EW from LW based on
# local intensity curvature; then segment_latewood optionally refines the LW
# mask by retaining only geometrically band-shaped LW regions.
#
# After running this section, inspect the three-panel figure (5c) to decide
# whether the spatial segmentation is sufficient.  If not, set USE_GMM = True
# in the Configuration section and re-run from there.

# --- 5a. findEWLW_distance_filtered ---

def findEWLW_distance_filtered(img, mask_wood, distance_thresh=3,
                                hessian_sigmas=None):
    """
    Segment a 3-D CT wood volume into earlywood (EW) and latewood (LW) using
    the Hessian curvature method combined with a distance-transform boundary
    filter.

    The normalised image is passed through DIPlib's Hessian operator at the
    specified scale(s).  The trace of the eigenvalue image (H_E) encodes
    local curvature: H_E < 0 → concave regions (EW, label 1);
    H_E ≥ 0 → convex regions (LW, label 2).  Only voxels at least
    `distance_thresh` pixels inside the wood boundary are considered.

    Parameters
    ----------
    img             : dip.Image or np.ndarray — 3-D calibrated CT volume
    mask_wood       : np.ndarray (bool)       — True where wood is present
    distance_thresh : int                     — minimum interior distance (px)
    hessian_sigmas  : list[float] or None     — Gaussian sigma(s) for Hessian;
                                                None uses DIPlib's default

    Returns
    -------
    labels_ewlw : np.ndarray (uint8) — 0 = background, 1 = EW, 2 = LW
    r_ew        : float              — EW volume fraction (within valid mask)
    r_lw        : float              — LW volume fraction (within valid mask)
    mask_valid  : np.ndarray (bool)  — voxels included in the segmentation
    """
    if hessian_sigmas is None:
        hessian_sigmas = []

    imgn = img - np.min(img)
    imgn = imgn / np.max(imgn)

    H              = dip.Hessian(imgn, sigmas=hessian_sigmas)
    eigenvalues, _ = dip.EigenDecomposition(H)
    H_E            = np.array(dip.Trace(eigenvalues))

    dist_inside = distance_transform_edt(mask_wood)
    mask_core   = dist_inside > distance_thresh
    mask_valid  = mask_core & mask_wood

    mask_ew = (H_E < 0)  & mask_valid
    mask_lw = (H_E >= 0) & mask_valid

    sum_valid = np.sum(mask_valid)
    r_ew = np.sum(mask_ew) / sum_valid if sum_valid > 0 else 0
    r_lw = np.sum(mask_lw) / sum_valid if sum_valid > 0 else 0

    if abs(r_ew + r_lw - 1) > 0.01:
        logging.warning("r_ew + r_lw = %.4f (expected ≈ 1.0)", r_ew + r_lw)

    labels_ewlw = np.zeros_like(img_data, dtype=np.uint8)
    labels_ewlw[mask_ew] = 1   # EW
    labels_ewlw[mask_lw] = 2   # LW

    return labels_ewlw, r_ew, r_lw, mask_valid


# Run primary segmentation on the calibrated volume
dip_img   = dip.Image(img_data_cal)
mask_wood = img_data_cal > 0

labels_ewlw, r_ew, r_lw, mask_valid = findEWLW_distance_filtered(
    dip_img, mask_wood,
    distance_thresh=DISTANCE_THRESH,
    hessian_sigmas=HESSIAN_SIGMAS
)
print(f"Primary segmentation — EW: {r_ew*100:.2f} %  |  LW: {r_lw*100:.2f} %")

# --- 5b. segment_latewood (geometric LW refinement) ---

def segment_latewood(labels_ewlw, min_area=10, border_tolerance=5,
                     min_ratio=1.5):
    """
    Refine the LW segmentation by retaining only geometrically band-shaped
    LW regions that plausibly represent true latewood bands.

    A connected LW component is kept if all three criteria are met:
        (a) area ≥ min_area pixels,
        (b) width-to-height ratio ≥ min_ratio,
        (c) the component contacts at least two of three reference edges
            (left, right, or bottom boundary of the wood section).
    Components that fail any criterion are reclassified as EW (label 1).

    Parameters
    ----------
    labels_ewlw      : np.ndarray (uint8, X×Y×Z) — 0=BG, 1=EW, 2=LW
    min_area         : int   — minimum area (px²)
    border_tolerance : int   — edge-contact tolerance (px)
    min_ratio        : float — minimum width / height ratio

    Returns
    -------
    labels_lw : np.ndarray (uint8, X×Y×Z) — refined segmentation
    """
    labels_lw = np.copy(labels_ewlw)
    _, _, shape_z = labels_ewlw.shape

    for z in range(shape_z):
        slice_labels  = labels_ewlw[:, :, z]
        labeled_slice = label(slice_labels == 2, connectivity=2)

        new_slice = np.ones_like(slice_labels)       # default: EW
        new_slice[slice_labels == 0] = 0             # restore background

        nonzero = np.where(slice_labels > 0)
        if len(nonzero[0]) == 0:
            labels_lw[:, :, z] = new_slice
            continue

        x_min = np.min(nonzero[1])
        x_max = np.max(nonzero[1])
        y_max = np.max(nonzero[0])

        for region in regionprops(labeled_slice):
            coords   = region.coords
            x_coords = coords[:, 1]
            y_coords = coords[:, 0]

            width  = np.max(x_coords) - np.min(x_coords)
            height = np.max(y_coords) - np.min(y_coords)
            ratio  = (width / height) if height > 0 else 0

            touches_left   = np.any(x_coords <= x_min + border_tolerance)
            touches_right  = np.any(x_coords >= x_max - border_tolerance)
            touches_bottom = np.any(y_coords >= y_max - border_tolerance)

            is_valid = (
                region.area >= min_area and
                ratio       >= min_ratio and
                (
                    (touches_left  and touches_right) or
                    (touches_left  and touches_bottom) or
                    (touches_right and touches_bottom)
                )
            )
            if is_valid:
                for y, x in coords:
                    new_slice[y, x] = 2

        labels_lw[:, :, z] = new_slice

    return labels_lw


labels_final = segment_latewood(
    labels_ewlw,
    min_area=LW_MIN_AREA,
    border_tolerance=LW_BORDER_TOL,
    min_ratio=LW_MIN_RATIO
)
print("segment_latewood refinement complete.")

# --- 5c. Visual quality check ---
# Inspect the three panels below.  If the refined segmentation (right panel)
# clearly separates EW and LW bands, set USE_GMM = False (already the
# default).  If the result shows scattered, non-band-shaped LW regions —
# typical for diffuse-porous hardwoods — set USE_GMM = True in the
# Configuration section and re-run from Section 6.

cmap_seg = ListedColormap(CUSTOM_COLORS)
norm_seg  = BoundaryNorm([0, 1, 2, 3], cmap_seg.N)

slice_ct      = img_data   [:, :, SLICE_PREVIEW]
slice_ewlw    = labels_ewlw[:, :, SLICE_PREVIEW]
slice_refined = labels_final[:, :, SLICE_PREVIEW]

fig, axs = plt.subplots(1, 3, figsize=(18, 6))

axs[0].imshow(slice_ct, cmap='gray', origin='lower', interpolation='none')
axs[0].set_title(f"Original CT (Z = {SLICE_PREVIEW})")
axs[0].set_xlabel("X [px]")
axs[0].set_ylabel("Y [px]")
axs[0].set_aspect('equal')

im1 = axs[1].imshow(slice_ewlw, cmap=cmap_seg, norm=norm_seg,
                    origin='lower', interpolation='none')
axs[1].set_title("Initial Segmentation (findEWLW)")
axs[1].set_xlabel("X [px]")
axs[1].set_aspect('equal')

im2 = axs[2].imshow(slice_refined, cmap=cmap_seg, norm=norm_seg,
                    origin='lower', interpolation='none')
axs[2].set_title("Refined Segmentation (segment_latewood)")
axs[2].set_xlabel("X [px]")
axs[2].set_aspect('equal')

cbar = fig.colorbar(im2, ax=axs, ticks=[0.5, 1.5, 2.5],
                    fraction=0.025, pad=0.04)
cbar.ax.set_yticklabels(['0 = Background', '1 = EW', '2 = LW'])
cbar.set_label("Class")
plt.tight_layout()
plt.show()

print("\n>>> Inspect the figure above.")
print("    If segmentation is visually sufficient → keep USE_GMM = False.")
print("    If EW/LW bands are not clearly separated → set USE_GMM = True")
print("    in the Configuration section and continue from Section 6.")


# =============================================================================
# 6. [OPTIONAL] GMM-BASED EW/LW DENSITY THRESHOLD
# =============================================================================
# Executed only when USE_GMM = True (see Configuration section).
#
# Background:
#   For diffuse-porous hardwoods (beech, oak), the Hessian-based spatial
#   segmentation often cannot reliably delineate EW and LW because the
#   anatomical density contrast between the two zones is much lower than in
#   conifers.  As an alternative, a Gaussian Mixture Model (GMM) is fitted
#   to the volumetric density histogram.  The histogram is assumed to consist
#   of two overlapping Gaussian populations — one for EW (lower density) and
#   one for LW (higher density).  Two candidate thresholds are computed:
#
#     • Midpoint         — arithmetic mean of the two Gaussian centres.
#                          More robust when the Gaussians overlap strongly.
#     • Gaussian intersection — algebraic intersection of the two weighted
#                          Gaussian curves.  More precise when the peaks are
#                          well separated.
#
#   The active threshold is selected via GMM_THRESHOLD_MODE in the
#   Configuration section.  All voxels below the threshold are assigned to
#   EW; all voxels at or above it are assigned to LW.

def gaussian_pdf(x, mean, std):
    """Evaluate a Gaussian probability density function."""
    return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))


def gaussian_intersection(w1, m1, s1, w2, m2, s2):
    """
    Compute the density value at which two weighted Gaussians intersect.

    Solves w1·N(x|m1,s1) = w2·N(x|m2,s2) analytically.  When the
    discriminant is negative (heavily overlapping Gaussians) it is clamped
    to zero.  The root closest to the interval [m1, m2] is returned.

    Parameters
    ----------
    w1, w2 : float — mixture weights
    m1, m2 : float — Gaussian means  (m1 < m2 assumed)
    s1, s2 : float — Gaussian standard deviations

    Returns
    -------
    float — threshold value between the two Gaussian centres
    """
    a = 1 / (2 * s1 ** 2) - 1 / (2 * s2 ** 2)
    b = m2 / (s2 ** 2) - m1 / (s1 ** 2)
    c = (m1 ** 2) / (2 * s1 ** 2) - (m2 ** 2) / (2 * s2 ** 2) + \
        np.log((s2 * w1) / (s1 * w2))

    if np.isclose(a, 0.0):
        return -c / b

    disc = max(b * b - 4 * a * c, 0.0)
    r1 = (-b + np.sqrt(disc)) / (2 * a)
    r2 = (-b - np.sqrt(disc)) / (2 * a)

    lo, hi = sorted([m1, m2])
    for r in (r1, r2):
        if lo <= r <= hi:
            return r
    mid = (lo + hi) / 2
    return r1 if abs(r1 - mid) < abs(r2 - mid) else r2


if USE_GMM:
    print("USE_GMM = True — running GMM-based density thresholding...")

    # Fit a 2-component GMM to the calibrated, background-excluded voxels
    gmm = GaussianMixture(n_components=2, covariance_type='full',
                          random_state=0)
    gmm.fit(values_all.reshape(-1, 1))

    w  = gmm.weights_
    mu = gmm.means_.ravel()
    sd = np.sqrt(gmm.covariances_.ravel())

    # Sort by mean so that component 0 = EW (lower density)
    idx  = np.argsort(mu)
    w1, w2 = w[idx[0]],  w[idx[1]]
    m1, m2 = mu[idx[0]], mu[idx[1]]
    s1, s2 = sd[idx[0]], sd[idx[1]]

    midpoint     = 0.5 * (m1 + m2)
    intersection = gaussian_intersection(w1, m1, s1, w2, m2, s2)

    print(f"GMM — EW: μ = {m1:.1f}, σ = {s1:.1f},  weight = {w1:.3f}")
    print(f"GMM — LW: μ = {m2:.1f}, σ = {s2:.1f},  weight = {w2:.3f}")
    print(f"Midpoint threshold       : {midpoint:.2f} kg/m³")
    print(f"Gaussian intersection    : {intersection:.2f} kg/m³")

    gmm_threshold = midpoint if GMM_THRESHOLD_MODE == 'midpoint' else intersection
    print(f"Active threshold ({GMM_THRESHOLD_MODE}): {gmm_threshold:.2f} kg/m³")

    # Visualise GMM fit on density histogram
    bins_gmm   = 200
    counts, edges = np.histogram(values_all, bins=bins_gmm)
    centers    = (edges[:-1] + edges[1:]) / 2
    bin_w      = edges[1] - edges[0]
    N          = values_all.size
    pdf_scale  = N * bin_w

    plt.figure(figsize=(12, 6))
    plt.bar(centers, counts, width=bin_w, color='lightgray',
            edgecolor='black', label='Histogram')
    plt.plot(centers, pdf_scale * w1 * gaussian_pdf(centers, m1, s1),
             label=f"Gaussian EW  (μ={m1:.1f}, σ={s1:.1f})")
    plt.plot(centers, pdf_scale * w2 * gaussian_pdf(centers, m2, s2),
             label=f"Gaussian LW  (μ={m2:.1f}, σ={s2:.1f})")
    plt.plot(centers,
             pdf_scale * (w1 * gaussian_pdf(centers, m1, s1) +
                          w2 * gaussian_pdf(centers, m2, s2)),
             linewidth=2, label="Gaussian mixture (EW + LW)")
    plt.axvline(midpoint,     color='darkred', linestyle='-',  linewidth=2.5,
                label=f"Midpoint: {midpoint:.1f} kg/m³")
    plt.axvline(intersection, color='red',     linestyle='--', linewidth=1.8,
                alpha=0.6,
                label=f"Intersection: {intersection:.1f} kg/m³")
    plt.title("Density Histogram with GMM-Based EW/LW Separation")
    plt.xlabel("Density [kg/m³]")
    plt.ylabel("Number of Voxels")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Separate EW and LW voxel values using the GMM threshold
    ew_values = values_all[values_all <  gmm_threshold]
    lw_values = values_all[values_all >= gmm_threshold]

    # Separate EW/LW histograms
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    c_ew, _, _ = axs[0].hist(ew_values, bins=200, color='purple',
                              edgecolor='purple')
    axs[0].set_title("Earlywood Density Histogram (GMM threshold)")
    axs[0].set_xlabel("Density [kg/m³]")
    axs[0].set_ylabel("Number of Voxels")
    axs[0].grid(True)

    c_lw, _, _ = axs[1].hist(lw_values, bins=200, color='orange',
                              edgecolor='orange')
    axs[1].set_title("Latewood Density Histogram (GMM threshold)")
    axs[1].set_xlabel("Density [kg/m³]")
    axs[1].set_ylabel("Number of Voxels")
    axs[1].grid(True)

    ymax = max(c_ew.max(), c_lw.max())
    axs[0].set_ylim(0, ymax)
    axs[1].set_ylim(0, ymax)
    plt.tight_layout()
    plt.show()

else:
    # Use spatial labels from the refined segmentation
    ew_values = img_data_cal[labels_final == 1].ravel()
    lw_values = img_data_cal[labels_final == 2].ravel()
    print(f"Using spatial segmentation labels — "
          f"EW voxels: {len(ew_values):,}, LW voxels: {len(lw_values):,}")


# =============================================================================
# 7. DENSITY STATISTICS
# =============================================================================

def compute_stats(values):
    """
    Compute descriptive density statistics for an array of voxel values.

    Returns a dict with: mean, mode, std, min, max, percentiles
    (P10/P25/P50/P75/P90), IQR, and bias-corrected Fisher kurtosis.
    """
    pct = np.percentile(values, [10, 25, 50, 75, 90])
    return {
        'mean':     np.mean(values),
        'mode':     mode(np.round(values, 1), keepdims=False)[0],
        'std':      np.std(values),
        'min':      np.min(values),
        'max':      np.max(values),
        'P10':      pct[0], 'P25': pct[1], 'P50': pct[2],
        'P75':      pct[3], 'P90': pct[4],
        'IQR':      pct[3] - pct[1],
        'kurtosis': kurtosis(values, fisher=True, bias=False)
    }


stats_all = compute_stats(values_all)
stats_ew  = compute_stats(ew_values)
stats_lw  = compute_stats(lw_values)


def print_stats(label_str, s):
    print(f"\n{label_str}")
    print(f"  Mean      : {s['mean']:.2f} kg/m³")
    print(f"  Mode      : {s['mode']:.2f} kg/m³")
    print(f"  Std Dev   : {s['std']:.2f}")
    print(f"  Min / Max : {s['min']:.2f} / {s['max']:.2f}")
    print(f"  Percentiles: P10={s['P10']:.2f}  P25={s['P25']:.2f}  "
          f"P50={s['P50']:.2f}  P75={s['P75']:.2f}  P90={s['P90']:.2f}")
    print(f"  IQR       : {s['IQR']:.2f}")
    print(f"  Kurtosis  : {s['kurtosis']:.3f}  (Fisher, bias-corrected)")


print_stats("Whole Sample", stats_all)
print_stats("Earlywood (EW)", stats_ew)
print_stats("Latewood  (LW)", stats_lw)

# Voxel coverage summary
total_vox    = len(values_all)
ew_vox       = len(ew_values)
lw_vox       = len(lw_values)
coverage_pct = (ew_vox + lw_vox) / total_vox * 100

print(f"\nVoxel counts — Total: {total_vox:,}  |  EW: {ew_vox:,}  "
      f"|  LW: {lw_vox:,}  |  Coverage: {coverage_pct:.2f} %")

# Comparative density histogram (whole sample, EW, LW)
fig, axs = plt.subplots(1, 2, figsize=(14, 7))

axs[0].hist(values_all, bins=200, color='gray', edgecolor='black')
axs[0].set_title("Whole-Sample Density Histogram")
axs[0].set_xlabel("Density [kg/m³]")
axs[0].set_ylabel("Number of Voxels")
axs[0].grid(True, linestyle='--', alpha=0.4)

axs[1].hist(ew_values, bins=100, alpha=0.6, color='purple',
            label=(f"EW  (μ={stats_ew['mean']:.1f}, "
                   f"σ={stats_ew['std']:.1f}, "
                   f"κ={stats_ew['kurtosis']:.2f})"))
axs[1].hist(lw_values, bins=100, alpha=0.6, color='orange',
            label=(f"LW  (μ={stats_lw['mean']:.1f}, "
                   f"σ={stats_lw['std']:.1f}, "
                   f"κ={stats_lw['kurtosis']:.2f})"))
axs[1].axvline(stats_ew['mean'], color='purple', linestyle='--')
axs[1].axvline(stats_lw['mean'], color='orange', linestyle='--')
axs[1].set_title("EW vs LW Density Histogram")
axs[1].set_xlabel("Density [kg/m³]")
axs[1].set_ylabel("Number of Voxels")
axs[1].legend()
axs[1].grid(True, linestyle='--', alpha=0.4)

ymax = max(axs[0].get_ylim()[1], axs[1].get_ylim()[1])
axs[0].set_ylim(0, ymax)
axs[1].set_ylim(0, ymax)
plt.tight_layout()
plt.show()


# =============================================================================
# 8. SAVE RESULTS TO CSV
# =============================================================================
# Results are appended to a shared CSV file (one row per sample).
# The header is written only when the file does not yet exist.

columns = [
    "Sample_ID",
    "Mean_all [kg/m^3]",  "Mode_all [kg/m^3]",  "Std_all [kg/m^3]",
    "Min_all [kg/m^3]",   "Max_all [kg/m^3]",
    "P10_all [kg/m^3]",   "P25_all [kg/m^3]",   "P50_all [kg/m^3]",
    "P75_all [kg/m^3]",   "P90_all [kg/m^3]",
    "IQR_all [kg/m^3]",   "Kurtosis_all [-]",
    "Mean_EW [kg/m^3]",   "Mode_EW [kg/m^3]",   "Std_EW [kg/m^3]",
    "Min_EW [kg/m^3]",    "Max_EW [kg/m^3]",
    "P10_EW [kg/m^3]",    "P25_EW [kg/m^3]",    "P50_EW [kg/m^3]",
    "P75_EW [kg/m^3]",    "P90_EW [kg/m^3]",
    "IQR_EW [kg/m^3]",    "Kurtosis_EW [-]",
    "Mean_LW [kg/m^3]",   "Mode_LW [kg/m^3]",   "Std_LW [kg/m^3]",
    "Min_LW [kg/m^3]",    "Max_LW [kg/m^3]",
    "P10_LW [kg/m^3]",    "P25_LW [kg/m^3]",    "P50_LW [kg/m^3]",
    "P75_LW [kg/m^3]",    "P90_LW [kg/m^3]",
    "IQR_LW [kg/m^3]",    "Kurtosis_LW [-]"
]

data_row = [
    SAMPLE_ID,
    stats_all['mean'], stats_all['mode'], stats_all['std'],
    stats_all['min'],  stats_all['max'],
    stats_all['P10'],  stats_all['P25'],  stats_all['P50'],
    stats_all['P75'],  stats_all['P90'],
    stats_all['IQR'],  stats_all['kurtosis'],
    stats_ew['mean'],  stats_ew['mode'],  stats_ew['std'],
    stats_ew['min'],   stats_ew['max'],
    stats_ew['P10'],   stats_ew['P25'],   stats_ew['P50'],
    stats_ew['P75'],   stats_ew['P90'],
    stats_ew['IQR'],   stats_ew['kurtosis'],
    stats_lw['mean'],  stats_lw['mode'],  stats_lw['std'],
    stats_lw['min'],   stats_lw['max'],
    stats_lw['P10'],   stats_lw['P25'],   stats_lw['P50'],
    stats_lw['P75'],   stats_lw['P90'],
    stats_lw['IQR'],   stats_lw['kurtosis']
]

df_out = pd.DataFrame([data_row], columns=columns)
os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
write_header = not os.path.exists(CSV_PATH)
df_out.to_csv(CSV_PATH, mode='a', index=False, header=write_header)
print(f"\nResults saved → {CSV_PATH}")

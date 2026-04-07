# =============================================================================
# STEP 2: EW/LW SEGMENTATION AND GROWTH-RING WIDTH MEASUREMENT
# CT Image Analysis — Master's Thesis
# =============================================================================
# Description:
#   This script segments the masked CT volume (output of Step 1) into
#   earlywood (EW) and latewood (LW) zones and computes per-ring widths.
#   The workflow consists of four sequential stages:
#
#     (1) Loading the masked CT volume and the original scan
#     (2) Primary segmentation via Hessian-based inter-mode analysis
#         (findEWLW_distance_filtered)
#     (3) [Optional] Secondary LW refinement via segment_latewood —
#         applied when primary segmentation produces insufficient results
#         for diffuse-porous species (e.g. beech, oak) where the EW/LW
#         boundary is anatomically less distinct
#     (4) 1-D radial profiling, run-length ring detection, and CSV export
#
# Environment: Python 3.13.1 | Jupyter Notebook (.ipynb) | VS Code
# Key packages: numpy, scipy, diplib, scikit-image, matplotlib, nrrd, pandas
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

from itertools import groupby
from scipy.ndimage import distance_transform_edt
from matplotlib.colors import ListedColormap, BoundaryNorm
from skimage.measure import label, regionprops


# =============================================================================
# 2. CONFIGURATION
# =============================================================================

# --- File paths ---
MASK_PATH   = r'C:\Users\jansi\OneDrive\Dokumenti\Master_thesis\CT_Data\MY_SAMPLES_Mask\Spruce\S1-30_MASK.nrrd'
SAMPLE_PATH = r'C:\Users\jansi\OneDrive\Dokumenti\Master_thesis\CT_Data\Samples_Master_thesis\Spruce\S1-30.nrrd'
SAVE_DIR    = r'C:\Users\jansi\OneDrive\Dokumenti\Master_thesis\CT_Data\EWLW_results\Spruce'
SAMPLE_ID   = 'S1-30_MASK'

# --- Primary segmentation parameters ---
DISTANCE_THRESH = 4     # Minimum distance (px) from wood boundary included in segmentation
                        # Use 4 for conifers (spruce/pine); use 3 for diffuse-porous species

# --- Secondary segmentation flag ---
# Set to True for diffuse-porous species (e.g. beech, oak) where primary
# segmentation alone produces insufficient EW/LW separation.
# Set to False for conifers (spruce, pine) where primary segmentation suffices.
USE_SEGMENT_LATEWOOD = False

# --- segment_latewood parameters (used only if USE_SEGMENT_LATEWOOD = True) ---
LW_MIN_AREA        = 40    # Minimum region area (px²) to be classified as LW
LW_BORDER_TOL      = 5     # Pixel tolerance when checking edge contact
LW_MIN_RATIO       = 1.5   # Minimum width/height ratio for a valid LW band

# --- 1-D profiling parameters ---
PIXEL_SIZE_MM = 0.3   # Voxel size in millimetres (from CT scanner settings)
PROFILE_OFFSET = 10   # Column offset (px) from centre for 3-column averaging

# --- Visualisation ---
SLICE_PREVIEW  = 150               # Default slice index (Z-axis) for previews
SLICE_INDICES  = [140, 200, 275]   # Slices shown in the multi-panel overview

# Colourmap: 0 = background (dark blue), 1 = EW (brown), 2 = LW (cyan)
CUSTOM_COLORS = ['#1f77b4', '#a0522d', '#00ced1']


# =============================================================================
# 3. LOAD CT DATA
# =============================================================================

# Masked volume (output from Step 1)
img_data, _ = nrrd.read(MASK_PATH)
print(f"Masked volume shape : {img_data.shape}")

# Original (unmasked) scan — used for visual reference panels
sample, _ = nrrd.read(SAMPLE_PATH)
print(f"Original scan shape : {sample.shape}")

# Visual check: mask and original side-by-side
fig, axs = plt.subplots(1, 2, figsize=(14, 7))
axs[0].imshow(img_data[:, :, SLICE_PREVIEW], cmap='gray', origin='lower')
axs[0].set_title(f"Masked CT Volume (Z = {SLICE_PREVIEW})")
axs[0].set_xlabel("X [px]")
axs[0].set_ylabel("Y [px]")

axs[1].imshow(sample[:, :, SLICE_PREVIEW], cmap='gray', origin='lower')
axs[1].set_title(f"Original CT Scan (Z = {SLICE_PREVIEW})")
axs[1].set_xlabel("X [px]")
axs[1].set_ylabel("Y [px]")

plt.tight_layout()
plt.show()


# =============================================================================
# 4. PRIMARY SEGMENTATION — findEWLW_distance_filtered
# =============================================================================

def findEWLW_distance_filtered(img, mask_wood, distance_thresh=4):
    """
    Segment a 3-D CT wood volume into earlywood (EW) and latewood (LW) using
    the Hessian curvature method combined with a distance-transform boundary
    filter.

    Method:
        The normalised image is passed to DIPlib's Hessian operator and its
        eigendecomposition.  The trace of the eigenvalue image (H_E) encodes
        local curvature: H_E < 0 corresponds to concave (EW) regions and
        H_E ≥ 0 to convex (LW) regions.  A distance transform ensures that
        only voxels at least `distance_thresh` pixels inside the wood boundary
        contribute to the segmentation, suppressing edge artefacts.

    Parameters
    ----------
    img             : dip.Image or np.ndarray — 3-D CT intensity volume
    mask_wood       : np.ndarray (bool)       — True where wood is present
    distance_thresh : int                     — minimum distance (px) from the
                                                wood boundary for a voxel to be
                                                included; higher values exclude
                                                more of the boundary zone

    Returns
    -------
    labels_ewlw : np.ndarray (uint8) — 0 = background, 1 = EW, 2 = LW
    r_ew        : float              — volume fraction of EW within valid mask
    r_lw        : float              — volume fraction of LW within valid mask
    mask_valid  : np.ndarray (bool)  — voxels used in the segmentation
    """
    # Normalise to [0, 1]
    imgn = img - np.min(img)
    imgn = imgn / np.max(imgn)

    # Hessian curvature analysis
    H              = dip.Hessian(imgn)
    eigenvalues, _ = dip.EigenDecomposition(H)
    H_E            = np.array(dip.Trace(eigenvalues))

    # Distance-transform boundary filter
    dist_inside = distance_transform_edt(mask_wood)
    mask_core   = dist_inside > distance_thresh
    mask_valid  = mask_core & mask_wood

    # Assign EW/LW labels
    mask_ew = (H_E < 0)  & mask_valid   # concave → EW (label 1)
    mask_lw = (H_E >= 0) & mask_valid   # convex  → LW (label 2)

    # Volume fractions
    sum_valid = np.sum(mask_valid)
    r_ew = np.sum(mask_ew) / sum_valid if sum_valid > 0 else 0
    r_lw = np.sum(mask_lw) / sum_valid if sum_valid > 0 else 0

    if abs(r_ew + r_lw - 1) > 0.01:
        logging.warning("r_ew + r_lw = %.4f (expected ≈ 1.0)", r_ew + r_lw)

    labels_ewlw = np.zeros_like(img_data, dtype=np.uint8)
    labels_ewlw[mask_ew] = 1   # EW
    labels_ewlw[mask_lw] = 2   # LW

    return labels_ewlw, r_ew, r_lw, mask_valid


# Run primary segmentation
dip_img  = dip.Image(img_data)
mask_wood = img_data > 0

labels_ewlw, r_ew, r_lw, mask_valid = findEWLW_distance_filtered(
    dip_img, mask_wood, distance_thresh=DISTANCE_THRESH
)

print(f"Primary segmentation complete.")
print(f"  Earlywood (EW) fraction : {r_ew * 100:.2f} %")
print(f"  Latewood  (LW) fraction : {r_lw * 100:.2f} %")


# =============================================================================
# 5. [OPTIONAL] SECONDARY SEGMENTATION — segment_latewood
# =============================================================================
# Applied only when USE_SEGMENT_LATEWOOD = True (see Configuration section).
#
# Background:
#   For diffuse-porous hardwood species (beech, oak), the density contrast
#   between EW and LW is substantially lower than in conifers.  The primary
#   Hessian-based segmentation can therefore misclassify medullary rays and
#   diffuse parenchyma as LW, producing scattered, non-band-shaped LW regions.
#   segment_latewood addresses this by retaining only those connected LW
#   components that span the expected transverse extent of a growth-ring band
#   (i.e., nearly the full width of the wood section) and discarding isolated
#   patches that do not meet geometric criteria.

def segment_latewood(labels_ewlw, min_area=40, border_tolerance=5,
                     min_ratio=1.5):
    """
    Refine the LW segmentation by retaining only geometrically band-shaped
    LW regions that plausibly represent true latewood bands.

    A connected LW component is kept as LW if all three conditions hold:
        (a) its area ≥ min_area pixels,
        (b) its width-to-height ratio ≥ min_ratio (i.e. it is wider than tall),
        (c) it contacts at least two of the three reference edges
            (left boundary, right boundary, bottom boundary of the wood section).
    Components that fail any condition are reassigned to EW (label 1).

    Parameters
    ----------
    labels_ewlw      : np.ndarray (uint8, shape X×Y×Z)
                           Input segmentation: 0 = background, 1 = EW, 2 = LW
    min_area         : int   — minimum area (px²) for a valid LW component
    border_tolerance : int   — pixel tolerance when testing edge contact
    min_ratio        : float — minimum width / height ratio for a LW band

    Returns
    -------
    labels_lw : np.ndarray (uint8, shape X×Y×Z)
                    Refined segmentation with the same label convention.
    """
    labels_lw = np.copy(labels_ewlw)
    _, _, shape_z = labels_ewlw.shape

    for z in range(shape_z):
        slice_labels = labels_ewlw[:, :, z]
        mask_lw      = slice_labels == 2

        labeled_slice = label(mask_lw, connectivity=2)

        # Default: everything becomes EW; background is restored below
        new_slice = np.ones_like(slice_labels)
        new_slice[slice_labels == 0] = 0

        # Determine the valid (non-background) bounding box of this slice
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

            # Keep as LW only if both size and geometry criteria are met
            is_valid = (
                region.area >= min_area and
                ratio >= min_ratio and
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


# Apply secondary segmentation if configured
if USE_SEGMENT_LATEWOOD:
    print("Applying secondary LW refinement (segment_latewood)...")
    labels_final = segment_latewood(
        labels_ewlw,
        min_area=LW_MIN_AREA,
        border_tolerance=LW_BORDER_TOL,
        min_ratio=LW_MIN_RATIO
    )
    print("Refinement complete.")
else:
    labels_final = labels_ewlw   # use primary result directly


# =============================================================================
# 6. VISUALISATION
# =============================================================================

cmap_seg = ListedColormap(CUSTOM_COLORS)
bounds   = [0, 1, 2, 3]
norm_seg = BoundaryNorm(bounds, cmap_seg.N)

# — 6a. Segmentation result vs original CT (single slice) —
slice_seg  = labels_final[:, :, SLICE_PREVIEW]
slice_orig = img_data[:,   :, SLICE_PREVIEW]

fig, axs = plt.subplots(1, 2, figsize=(14, 7))

im0 = axs[0].imshow(slice_seg, cmap=cmap_seg, norm=norm_seg,
                    origin='lower', interpolation='none')
axs[0].set_title(f"EW / LW Segmentation (Z = {SLICE_PREVIEW})")
axs[0].set_xlabel("X [px]")
axs[0].set_ylabel("Y [px]")
cbar = fig.colorbar(im0, ax=axs[0], ticks=[0.5, 1.5, 2.5])
cbar.ax.set_yticklabels(['0 = Background', '1 = EW', '2 = LW'])

axs[1].imshow(slice_orig, cmap='gray', origin='lower', interpolation='none')
axs[1].set_title(f"Original CT Slice (Z = {SLICE_PREVIEW})")
axs[1].set_xlabel("X [px]")
axs[1].set_ylabel("Y [px]")

plt.tight_layout()
plt.show()

# — 6b. Comparison: initial vs refined segmentation (only if refinement applied) —
if USE_SEGMENT_LATEWOOD:
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    axs[0].imshow(slice_orig, cmap='gray', origin='lower', interpolation='none')
    axs[0].set_title(f"Original CT (Z = {SLICE_PREVIEW})")
    axs[0].set_xlabel("X [px]")
    axs[0].set_ylabel("Y [px]")

    im1 = axs[1].imshow(labels_ewlw[:, :, SLICE_PREVIEW], cmap=cmap_seg,
                        norm=norm_seg, origin='lower', interpolation='none')
    axs[1].set_title("Initial Segmentation (findEWLW)")
    axs[1].set_xlabel("X [px]")

    im2 = axs[2].imshow(slice_seg, cmap=cmap_seg, norm=norm_seg,
                        origin='lower', interpolation='none')
    axs[2].set_title("Refined Segmentation (segment_latewood)")
    axs[2].set_xlabel("X [px]")

    cbar = fig.colorbar(im2, ax=axs, ticks=[0.5, 1.5, 2.5],
                        fraction=0.034, pad=0.04)
    cbar.ax.set_yticklabels(['0 = Background', '1 = EW', '2 = LW'])
    cbar.set_label("Class")

    plt.tight_layout()
    plt.show()

# — 6c. Multi-slice overview —
fig, axs = plt.subplots(1, len(SLICE_INDICES), figsize=(18, 6))
for i, z in enumerate(SLICE_INDICES):
    im = axs[i].imshow(labels_final[:, :, z], cmap=cmap_seg, norm=norm_seg,
                       origin='lower', interpolation='none')
    axs[i].set_title(f"EW/LW Segmentation (Z = {z})")
    axs[i].set_xlabel("X [px]")
    axs[i].set_ylabel("Y [px]")
cbar = fig.colorbar(im, ax=axs, ticks=[0.5, 1.5, 2.5], fraction=0.035, pad=0.04)
cbar.ax.set_yticklabels(['0 = Background', '1 = EW', '2 = LW'])
cbar.set_label("Class")
plt.tight_layout()
plt.show()

# — 6d. EW vs LW intensity histogram (volumetric) —
ew_vals = img_data[labels_final == 1].ravel()
lw_vals = img_data[labels_final == 2].ravel()

ew_mean, ew_std = ew_vals.mean(), ew_vals.std()
lw_mean, lw_std = lw_vals.mean(), lw_vals.std()
print(f"EW — mean = {ew_mean:.1f} HU,  std = {ew_std:.1f} HU")
print(f"LW — mean = {lw_mean:.1f} HU,  std = {lw_std:.1f} HU")

plt.figure(figsize=(8, 6))
plt.hist(ew_vals, bins=100, color='purple', alpha=0.5,
         label=f'EW  (μ = {ew_mean:.1f},  σ = {ew_std:.1f})')
plt.hist(lw_vals, bins=100, color='orange', alpha=0.5,
         label=f'LW  (μ = {lw_mean:.1f},  σ = {lw_std:.1f})')
plt.axvline(ew_mean, color='purple', linestyle='--', linewidth=1)
plt.axvline(lw_mean, color='orange', linestyle='--', linewidth=1)
plt.title("CT Intensity Histogram: Earlywood vs Latewood")
plt.xlabel("CT Grey Value (HU)")
plt.ylabel("Voxel Count")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()


# =============================================================================
# 7. 1-D RADIAL PROFILE AND GROWTH-RING DETECTION
# =============================================================================
# Growth-ring widths are measured on a representative slice using a 1-D
# intensity profile averaged across three central columns (centre ± offset).
# Consecutive EW→LW run pairs in the profile define individual growth rings.

# — 7a. Extract 1-D EW/LW label profile —
seg_slice = labels_final[:, :, SLICE_PREVIEW]
ct_slice  = img_data[:,   :, SLICE_PREVIEW]

x_sum    = np.sum(seg_slice > 0, axis=0)
x_coords = np.where(x_sum > 0)[0]
x_min, x_max = x_coords[0], x_coords[-1]

x_middle = (x_min + x_max) // 2
x_left   = max(x_min, x_middle - PROFILE_OFFSET)
x_right  = min(x_max, x_middle + PROFILE_OFFSET)

avg_line     = (seg_slice[:, x_left] +
                seg_slice[:, x_middle] +
                seg_slice[:, x_right]) / 3.0
rounded_line = np.round(avg_line).astype(int)

# — 7b. Visualise profile with sampling columns —
cmap_profile = ListedColormap(["black", "purple", "orange"])

fig, axs = plt.subplots(1, 3, figsize=(18, 6))

axs[0].plot(rounded_line, linewidth=1.5)
axs[0].set_title("Average EW/LW Label Profile (3 columns)")
axs[0].set_xlabel("Y (row index)")
axs[0].set_ylabel("Label  (0 = BG, 1 = EW, 2 = LW)")
axs[0].legend([f"Columns: {x_left}, {x_middle}, {x_right}  (±{PROFILE_OFFSET} px)"])
axs[0].grid(True)

im1 = axs[1].imshow(seg_slice, cmap=cmap_profile, origin='lower')
axs[1].axvline(x_left,   color='red',   linestyle='--', label=f"X = {x_left}")
axs[1].axvline(x_middle, color='green', linestyle='--', label=f"X = {x_middle}")
axs[1].axvline(x_right,  color='blue',  linestyle='--', label=f"X = {x_right}")
axs[1].set_title(f"EW/LW Segmentation with Sampling Columns (Z = {SLICE_PREVIEW})")
axs[1].set_xlabel("X [px]")
axs[1].set_ylabel("Y [px]")
axs[1].legend()
cbar = plt.colorbar(im1, ax=axs[1], ticks=[0, 1, 2])
cbar.ax.set_yticklabels(['0 = Background', '1 = EW', '2 = LW'])

axs[2].imshow(ct_slice, cmap='gray', origin='lower')
axs[2].axvline(x_left,   color='red',   linestyle='--')
axs[2].axvline(x_middle, color='green', linestyle='--')
axs[2].axvline(x_right,  color='blue',  linestyle='--')
axs[2].set_title(f"Original CT Slice {SLICE_PREVIEW} with Sampling Columns")
axs[2].set_xlabel("X [px]")
axs[2].set_ylabel("Y [px]")

plt.tight_layout()
plt.show()

# — 7c. Run-length encoding and ring pairing —
runs  = []
start = 0
for lbl, group in groupby(rounded_line):
    length = sum(1 for _ in group)
    runs.append((lbl, start, length))
    start += length

# Discard background segments
runs = [(lbl, s, l) for lbl, s, l in runs if lbl > 0]

# Pair consecutive EW (1) → LW (2) segments into growth rings
all_ew_pix, all_lw_pix, all_total_pix = [], [], []
growth_rings = []

for (lab1, s1, l1), (lab2, s2, l2) in zip(runs, runs[1:]):
    if lab1 == 1 and lab2 == 2:
        all_ew_pix.append(l1)
        all_lw_pix.append(l2)
        all_total_pix.append(l1 + l2)
        growth_rings.append({
            'ew_start_px': s1,       'ew_end_px': s1 + l1 - 1,
            'lw_start_px': s2,       'lw_end_px': s2 + l2 - 1
        })

# — 7d. Convert pixel counts to millimetres —
all_ew_mm    = [w * PIXEL_SIZE_MM for w in all_ew_pix]
all_lw_mm    = [w * PIXEL_SIZE_MM for w in all_lw_pix]
all_total_mm = [w * PIXEL_SIZE_MM for w in all_total_pix]

print(f"\nDetected {len(all_ew_mm)} growth rings")
print(f"Earlywood widths  [mm]:  mean={np.mean(all_ew_mm):.3f},  "
      f"std={np.std(all_ew_mm):.3f},  "
      f"min={np.min(all_ew_mm):.3f},  max={np.max(all_ew_mm):.3f}")
print(f"Latewood  widths  [mm]:  mean={np.mean(all_lw_mm):.3f},  "
      f"std={np.std(all_lw_mm):.3f},  "
      f"min={np.min(all_lw_mm):.3f},  max={np.max(all_lw_mm):.3f}")
print(f"Total ring widths [mm]:  mean={np.mean(all_total_mm):.3f},  "
      f"std={np.std(all_total_mm):.3f},  "
      f"min={np.min(all_total_mm):.3f},  max={np.max(all_total_mm):.3f}")

# Per-ring table
df_rings = pd.DataFrame({
    'ring_index': range(1, len(all_ew_mm) + 1),
    'EW_mm':      all_ew_mm,
    'LW_mm':      all_lw_mm,
    'Total_mm':   all_total_mm
})
display(df_rings)


# =============================================================================
# 8. SAVE RESULTS TO CSV
# =============================================================================

# Summary statistics table
stats_df = pd.DataFrame({
    'mean': df_rings[['EW_mm', 'LW_mm', 'Total_mm']].mean(),
    'std':  df_rings[['EW_mm', 'LW_mm', 'Total_mm']].std(),
    'min':  df_rings[['EW_mm', 'LW_mm', 'Total_mm']].min(),
    'max':  df_rings[['EW_mm', 'LW_mm', 'Total_mm']].max()
}).T
stats_df.index.name = 'statistic'

# Write per-ring data + summary to a single CSV
os.makedirs(SAVE_DIR, exist_ok=True)
out_path = os.path.join(SAVE_DIR, f"{SAMPLE_ID}_ring_widths.csv")

with open(out_path, 'w', newline='') as f:
    df_rings.to_csv(f, index=False)
    f.write('\n')       # blank separator line
    stats_df.to_csv(f)

print(f"\nRing width data and summary saved → {out_path}")

# =============================================================================
# STEP 1: BINARY WOOD MASK CREATION
# CT Image Analysis — Master's Thesis
# =============================================================================
# Description:
#   This script processes raw CT scan data (.nrrd format) to produce a binary
#   wood mask that isolates the wood volume from the scan background. The
#   workflow consists of four sequential stages:
#     (1) Loading the 3-D CT volume
#     (2) Inter-mode histogram thresholding to separate wood from air
#     (3) Morphological operations to fill holes and refine the mask boundary
#     (4) Contour-based edge trimming applied across all slices
#   The resulting masked volume is saved as a new .nrrd file for use in
#   subsequent analysis steps.
#
# Environment: Python 3.13.1 | Jupyter Notebook (.ipynb) | VS Code
# Key packages: numpy, scipy, scikit-image, diplib, opencv-python, nrrd
# =============================================================================


# =============================================================================
# 1. IMPORTS
# =============================================================================

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import cv2
import nrrd
import diplib as dip
from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes


# =============================================================================
# 2. CONFIGURATION
# =============================================================================

# --- File paths ---
INPUT_PATH  = r'C:\Users\jansi\OneDrive\Dokumenti\Master_thesis\CT_Data\MY_SAMPLES\Big\B1_sample.nrrd'
OUTPUT_PATH = r'C:\Users\jansi\OneDrive\Dokumenti\Master_thesis\CT_Data\MY_SAMPLES_Mask\Big\B1_MASK1_Sample.nrrd'

# --- Thresholding parameters ---
RHO_MIN = 200   # Lower HU bound for inter-mode thresholding
RHO_MAX = 430   # Upper HU bound for inter-mode thresholding

# --- Morphological operation parameters ---
DILATION_ITERATIONS = 4   # Number of dilation iterations
EROSION_ITERATIONS  = 5   # Number of erosion iterations

# --- Contour trimming parameters ---
X_OFFSET         = 3    # Pixels trimmed from left/right edges of bounding box
Y_OFFSET         = 2    # Pixels trimmed from top/bottom edges of bounding box
MIN_CONTOUR_AREA = 100  # Minimum contour area (px²) — smaller contours are skipped

# --- Visualisation ---
SLICE_PREVIEW = 150   # Slice index (Z-axis) used for all intermediate previews


# =============================================================================
# 3. LOAD CT DATA
# =============================================================================

img, header = nrrd.read(INPUT_PATH)
print(f"Loaded CT volume — shape: {img.shape}")   # expected: (rows, cols, depth)

# Visual check: raw cross-section before any processing
plt.figure(figsize=(6, 5))
plt.imshow(img[:, :, SLICE_PREVIEW], cmap='gray')
plt.colorbar()
plt.title(f"Raw CT Cross-Section (Z = {SLICE_PREVIEW})")
plt.tight_layout()
plt.show()


# =============================================================================
# 4. INTER-MODE THRESHOLDING
# =============================================================================

def find_inter_mode(img, rho_min, rho_max):
    """
    Locate the threshold between two histogram modes within a given HU range.

    The pixel intensities inside [rho_min, rho_max] are histogrammed, the
    histogram is Gaussian-smoothed (sigma = 3) using DIPlib, and the HU value
    at the smoothed histogram minimum is returned as the threshold.

    Parameters
    ----------
    img     : np.ndarray  — 2-D or 3-D array of HU values
    rho_min : int         — lower bound of the HU range of interest
    rho_max : int         — upper bound of the HU range of interest

    Returns
    -------
    threshold : float        — HU value at the inter-mode histogram minimum
    smoothed  : dip.Image    — Gaussian-smoothed histogram frequencies
    bin_edges : np.ndarray   — bin edges of the computed histogram
    """
    mask = (img >= rho_min) & (img <= rho_max)
    if not np.any(mask):
        logging.warning("No pixels found in HU range [%d, %d].", rho_min, rho_max)
        return None, None, None

    rhos      = img[mask]
    nbins     = rho_max - rho_min + 1
    freq, bin_edges = np.histogram(rhos, bins=nbins)
    smoothed  = dip.Gauss(freq, 3)

    threshold = bin_edges[np.argmin(smoothed)]
    return threshold, smoothed, bin_edges


# Apply thresholding on the preview slice
slice_img = img[:, :, SLICE_PREVIEW]
threshold, smoothed_freq, bin_edges = find_inter_mode(slice_img, RHO_MIN, RHO_MAX)
print(f"Inter-mode threshold: {threshold:.1f} HU")

# Visualise smoothed histogram with threshold marker
plt.figure(figsize=(10, 5))
plt.plot(bin_edges[:-1], smoothed_freq, color='steelblue', label="Smoothed Histogram")
plt.axvline(threshold, color='red', linestyle='--',
            label=f"Threshold = {threshold:.1f} HU")
plt.title(f"Inter-mode Thresholding Histogram (Z = {SLICE_PREVIEW})")
plt.xlabel("HU Value")
plt.ylabel("Smoothed Frequency")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# =============================================================================
# 5. BINARY MASK — MORPHOLOGICAL OPERATIONS
# =============================================================================

# --- 5a. Initial binary mask from threshold ---
mask = img >= threshold

# --- 5b. Fill internal holes slice by slice (XY plane) ---
mask = np.vectorize(binary_fill_holes, signature='(n,m)->(n,m)')(mask)

# --- 5c. Dilation — connect near-boundary wood pixels ---
mask_new = binary_dilation(mask, iterations=DILATION_ITERATIONS)

# --- 5d. Erosion — restore approximate original boundary after dilation ---
mask_new = binary_erosion(mask_new, iterations=EROSION_ITERATIONS)

# Visual check after morphological operations
plt.figure(figsize=(6, 5))
plt.imshow(mask_new[:, :, SLICE_PREVIEW], cmap='gray')
plt.title(f"Mask after Morphological Operations (Z = {SLICE_PREVIEW})")
plt.axis('on')
plt.tight_layout()
plt.show()


# =============================================================================
# 6. CONTOUR-BASED BOUNDARY TRIMMING (ALL SLICES)
# =============================================================================
# For each Z-slice the largest external contour is detected, its bounding box
# is computed, inset by the configured offsets, and only the pixels within
# that trimmed rectangle are retained.  This removes residual edge artefacts
# that survive morphological processing.

depth        = mask_new.shape[2]
mask_cleaned = np.zeros_like(mask_new, dtype=np.uint8)

for z in range(depth):
    slice_2d = mask_new[:, :, z].astype(np.uint8)

    contours, _ = cv2.findContours(
        slice_2d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        continue   # skip slices with no detected contours

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
        continue   # skip artefact-sized regions

    # Simplify contour and extract bounding coordinates
    epsilon  = 0.01 * cv2.arcLength(contour, True)
    approx   = cv2.approxPolyDP(contour, epsilon, True)
    x_coords = approx[:, 0, 0]
    y_coords = approx[:, 0, 1]

    x_left   = max(int(np.min(x_coords)) + X_OFFSET, 0)
    x_right  = min(int(np.max(x_coords)) - X_OFFSET, slice_2d.shape[1] - 1)
    y_top    = max(int(np.min(y_coords)) + Y_OFFSET, 0)
    y_bottom = min(int(np.max(y_coords)) - Y_OFFSET, slice_2d.shape[0] - 1)

    # Retain pixels inside the trimmed bounding box only
    cleaned = np.zeros_like(slice_2d)
    cleaned[y_top:y_bottom + 1, x_left:x_right + 1] = \
        slice_2d[y_top:y_bottom + 1, x_left:x_right + 1]
    mask_cleaned[:, :, z] = cleaned

# Visual check of the final cleaned mask
plt.figure(figsize=(6, 5))
plt.imshow(mask_cleaned[:, :, SLICE_PREVIEW], cmap='gray')
plt.title(f"Cleaned Mask (Z = {SLICE_PREVIEW})")
plt.axis('on')
plt.tight_layout()
plt.show()


# =============================================================================
# 7. APPLY MASK TO ORIGINAL VOLUME AND SAVE
# =============================================================================

img_masked = img * mask_cleaned
print(f"Masked volume shape: {img_masked.shape}")

# Final visual check
plt.figure(figsize=(6, 5))
plt.imshow(img_masked[:, :, SLICE_PREVIEW], cmap='gray')
plt.colorbar()
plt.title(f"Masked CT Cross-Section (Z = {SLICE_PREVIEW})")
plt.tight_layout()
plt.show()

# Save masked volume
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
nrrd.write(OUTPUT_PATH, img_masked)
print(f"Masked volume saved → {OUTPUT_PATH}")

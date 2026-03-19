"""
GFP / DAPI - Percent Positive Cell Quantification
=================================================
Processes merged RGBA TIF images where blue = DAPI and green = GFP/FITC.

Usage:
    pip install -r requirements.txt
    python gfp_dapi_counter.py

Edit the settings below before running.
"""

import csv
import os

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage
from skimage import filters, measure, morphology, segmentation
from skimage.feature import peak_local_max

# ===================== SETTINGS - EDIT THESE =====================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# Name of your secondary-only control image (must match exactly)
CONTROL_IMAGE = "Heptocytes_FOXO1A+_121025_FITC_20x_2ndaryonly.tif"

# Threshold multiplier: GFP threshold = control_mean + MULT * control_SD
THRESHOLD_MULT = 3.0

# Gaussian blur sigma (reduces noise before thresholding)
BLUR_SIGMA = 1.0

# Crop bottom N pixels to remove scale bar (set to 0 to disable)
CROP_BOTTOM = 40

# Nucleus detection (DAPI)
MIN_NUCLEUS_SIZE = 30   # minimum nucleus area in pixels
MAX_NUCLEUS_SIZE = 5000  # maximum nucleus area in pixels
WATERSHED_MIN_DIST = 10  # minimum distance between nuclei peaks for watershed

# GFP object detection
MIN_GFP_SIZE = 30   # minimum GFP object area in pixels
MAX_GFP_SIZE = 5000
MIN_GFP_INTENSITY = 45  # absolute minimum GFP intensity (0-255) to count as positive
                         # This acts as a floor - even if the control threshold is lower,
                         # GFP signal below this value is considered background.
                         # Increase to be more strict (e.g. 100, 125, 150)

# ===================== END SETTINGS ==============================


def gaussian_blur(image, sigma):
    """Apply Gaussian blur to a 2D image."""
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(image.astype(float), sigma=sigma)


def count_nuclei(blue_channel, blur_sigma, min_size, max_size, ws_min_dist):
    """Count DAPI nuclei using Yen threshold plus watershed."""
    blurred = gaussian_blur(blue_channel, blur_sigma)
    thresh = filters.threshold_yen(blurred.astype(np.uint8))

    mask = blurred > thresh
    mask = ndimage.binary_fill_holes(mask)
    mask = morphology.remove_small_objects(mask, min_size=max(1, min_size - 1))

    # Watershed to split touching nuclei
    distance = ndimage.distance_transform_edt(mask)
    coords = peak_local_max(distance, min_distance=ws_min_dist, labels=mask)
    markers_mask = np.zeros(distance.shape, dtype=bool)
    if len(coords) > 0:
        markers_mask[tuple(coords.T)] = True
    markers = ndimage.label(markers_mask)[0]
    labels = segmentation.watershed(-distance, markers, mask=mask)

    # Filter by size
    props = measure.regionprops(labels)
    count = sum(1 for p in props if min_size <= p.area <= max_size)

    return count, mask, thresh


def count_gfp_positive(green_channel, threshold, blur_sigma, min_size, max_size, min_intensity):
    """Count GFP-positive objects above threshold and minimum mean intensity."""
    blurred = gaussian_blur(green_channel, blur_sigma)

    mask = blurred > threshold
    mask = ndimage.binary_fill_holes(mask)
    mask = morphology.remove_small_objects(mask, min_size=max(1, min_size - 1))

    labels = ndimage.label(mask)[0]
    props = measure.regionprops(labels, intensity_image=green_channel)

    # Filter by size and mean intensity - only count genuinely bright objects.
    count = sum(
        1
        for p in props
        if min_size <= p.area <= max_size and p.mean_intensity >= min_intensity
    )

    return count, mask


def main():
    print("=" * 56)
    print("  GFP/DAPI Positive Cell Quantification (Python)")
    print("=" * 56)

    output_dir = os.path.join(INPUT_DIR, "results")
    os.makedirs(output_dir, exist_ok=True)

    # Gather image files
    extensions = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
    image_files = sorted(
        [
            f
            for f in os.listdir(INPUT_DIR)
            if f.lower().endswith(extensions) and os.path.isfile(os.path.join(INPUT_DIR, f))
        ]
    )

    if len(image_files) == 0:
        print(f"ERROR: No image files found in {INPUT_DIR}")
        return

    print(f"\nFound {len(image_files)} images in {INPUT_DIR}")
    print(f"Control: {CONTROL_IMAGE}\n")

    # ----------------------------------------------------------
    #  STEP 1: Measure control to set GFP threshold
    # ----------------------------------------------------------
    ctrl_path = os.path.join(INPUT_DIR, CONTROL_IMAGE)
    if not os.path.exists(ctrl_path):
        print(f"ERROR: Control image not found: {ctrl_path}")
        return

    ctrl_img = tifffile.imread(ctrl_path)
    if CROP_BOTTOM > 0:
        ctrl_img = ctrl_img[:-CROP_BOTTOM, :, :]
    ctrl_green = ctrl_img[:, :, 1].astype(float)
    ctrl_green_blurred = gaussian_blur(ctrl_green, BLUR_SIGMA)

    ctrl_mean = ctrl_green_blurred.mean()
    ctrl_sd = ctrl_green_blurred.std()
    gfp_threshold_calc = ctrl_mean + THRESHOLD_MULT * ctrl_sd
    gfp_threshold = max(gfp_threshold_calc, MIN_GFP_INTENSITY)

    print(f">> CONTROL: {CONTROL_IMAGE}")
    print(f"   Green channel: mean={ctrl_mean:.2f}  SD={ctrl_sd:.2f}")
    print(f"   Control-based threshold: {gfp_threshold_calc:.1f}  (mean + {THRESHOLD_MULT}*SD)")
    print(f"   Min intensity floor:     {MIN_GFP_INTENSITY}")
    print(f"   Final GFP threshold:     {gfp_threshold:.1f}\n")

    # ----------------------------------------------------------
    #  STEP 2: Process all images
    # ----------------------------------------------------------
    results = []

    for fname in image_files:
        fpath = os.path.join(INPUT_DIR, fname)
        is_ctrl = fname == CONTROL_IMAGE
        tag = " [CONTROL]" if is_ctrl else ""

        print(f">> {fname}{tag}")

        img = tifffile.imread(fpath)

        # Crop bottom to remove scale bar
        if CROP_BOTTOM > 0 and img.ndim >= 2:
            img = img[:-CROP_BOTTOM, :] if img.ndim == 2 else img[:-CROP_BOTTOM, :, :]

        # Handle different channel layouts
        if img.ndim == 2:
            print("   SKIP: single-channel image\n")
            continue
        if img.shape[2] < 3:
            print("   SKIP: fewer than 3 channels\n")
            continue

        blue = img[:, :, 2]  # DAPI
        green = img[:, :, 1]  # GFP

        # Count DAPI nuclei
        total_nuclei, dapi_mask, yen_thresh = count_nuclei(
            blue, BLUR_SIGMA, MIN_NUCLEUS_SIZE, MAX_NUCLEUS_SIZE, WATERSHED_MIN_DIST
        )

        # Count GFP+ cells
        gfp_count, gfp_mask = count_gfp_positive(
            green, gfp_threshold, BLUR_SIGMA, MIN_GFP_SIZE, MAX_GFP_SIZE, MIN_GFP_INTENSITY
        )

        pct = (gfp_count / total_nuclei * 100) if total_nuclei > 0 else 0.0

        print(f"   DAPI threshold (Yen): {yen_thresh}")
        print(f"   DAPI nuclei:  {total_nuclei}")
        print(f"   GFP+ cells:   {gfp_count}")
        print(f"   % Positive:   {pct:.1f}%\n")

        sample_type = "2nd-Only Control" if is_ctrl else "Sample"
        results.append([fname, sample_type, total_nuclei, gfp_count, f"{pct:.1f}"])

        # Save masks
        base = fname.rsplit(".", 1)[0]
        Image.fromarray((dapi_mask * 255).astype(np.uint8)).save(
            os.path.join(output_dir, f"{base}_DAPI_mask.png")
        )
        Image.fromarray((gfp_mask * 255).astype(np.uint8)).save(
            os.path.join(output_dir, f"{base}_GFP_mask.png")
        )

    # ----------------------------------------------------------
    #  STEP 3: Save CSV
    # ----------------------------------------------------------
    csv_path = os.path.join(output_dir, "Positive_Cell_Results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Image", "Type", "Total Nuclei", "GFP+ Cells", "% Positive"])
        writer.writerows(results)

    print("=" * 56)
    print("  DONE - Results saved to:")
    print(f"  {csv_path}")
    print(f"  DAPI & GFP masks saved in: {output_dir}")
    print("=" * 56)

    # Print summary table
    print(f"\n{'Image':<55} {'Type':<18} {'Nuclei':>7} {'GFP+':>6} {'%Pos':>6}")
    print("-" * 95)
    for row in results:
        print(f"{row[0]:<55} {row[1]:<18} {row[2]:>7} {row[3]:>6} {row[4]:>5}%")


if __name__ == "__main__":
    main()

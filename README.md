# FOXO1A GFP Positive Cell Quantification

Python script for quantifying GFP-positive cells from merged fluorescence images using the DAPI channel for nuclei counting and the GFP/FITC channel for positive-cell detection.

## What It Does

- Uses the blue channel for DAPI nucleus segmentation.
- Uses the green channel for GFP/FITC signal detection.
- Sets the GFP threshold from a secondary-only control image.
- Exports binary masks and a CSV summary of percent-positive cells.

## Files

- `gfp_dapi_counter.py`: main analysis script
- `requirements.txt`: Python dependencies

## Usage

```bash
pip install -r requirements.txt
python gfp_dapi_counter.py
```

By default, the script looks for input images in the parent directory of the script folder. Update the settings at the top of the script if your data lives somewhere else.

## Expected Inputs

- Merged images with at least 3 channels
- Blue channel = DAPI
- Green channel = GFP/FITC
- A secondary-only control image matching the `CONTROL_IMAGE` setting


# IU-CXR Sanity Subset Creation Process

## Source Files
- indiana_reports.csv
- indiana_projections.csv
- images_normalized/

## Preprocessing Rules
1. `caption = indication + findings + impression`
2. Remove rows where:
   - MeSH contains "normal"
   - Problems contains "normal"
   - MeSH contains "Technical Quality of Image Unsatisfactory"
   - Problems contains "Technical Quality of Image Unsatisfactory"
3. Remove rows with missing images.
4. Normalize caption whitespace and punctuation.

## Pathology Label Extraction
Combined:
- MeSH
- Problems
- caption (free text)

to compute 14 CheXpert-like pathology labels tuned for IU dataset.

## Sampling
- Pathology-rich UIDs preferred.
- Technical-unsatisfactory and normal studies excluded before sampling.
- All views associated with selected UIDs included.

## Outputs
- sanity_subset_iucxr_v02.csv
- iu_cxr_technical_unsatisfactory.csv
- Per-image RAW/CLAHE panels
- UID mini-grids
- Collage for poster
- Sample metadata table

## Running the Sanity Subset Script
From the repository root, run:

```bash
python -m phase1.scripts.select_iucxr_subset
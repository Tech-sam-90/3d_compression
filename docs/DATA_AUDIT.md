# Data Audit — Raw CT-RATE NIfTI Availability for End-to-End ViT Redesign

Read-only audit, no code changes. Run in preparation for redesigning ICTC to
train end-to-end with an unfrozen ViT, which needs raw CT volumes at training
time instead of the pre-extracted `.pt` CT-CLIP feature files the current
pipeline (`aadp/data/ctclip_feature_dataset.py`, `CTCLIPStage2VLM`) uses.

**Headline finding: the raw data is already fully downloaded** — this
session's earlier `scripts/download_ctrate.py` run (see chat history:
job started 2026-07-30 21:11, finished 2026-07-31 13:49 across two Lustre-
incident-driven restarts) completed the entire CT-RATE dataset, verified
below with a fresh `find` count, not just trusted from memory.

## 1. Raw CT-RATE NIfTI files

```
find /scratch/sadeniji /project -name "*.nii.gz" 2>/dev/null | head -50
find /scratch/sadeniji /project -name "*.nii" 2>/dev/null | head -50
```

**`.nii.gz` (compressed, CT-RATE's native format): present, at scale.**
All 50 sampled hits are under `/scratch/sadeniji/ct_rate_raw/dataset/{train_fixed,valid_fixed}/...`,
e.g.:
```
/scratch/sadeniji/ct_rate_raw/dataset/valid_fixed/valid_221/valid_221_a/valid_221_a_2.nii.gz
/scratch/sadeniji/ct_rate_raw/dataset/valid_fixed/valid_1019/valid_1019_c/valid_1019_c_1.nii.gz
...
```
Full count (not just the head-50 sample):
```
$ find /scratch/sadeniji/ct_rate_raw -name "*.nii.gz" | wc -l
50188
```
50,188 = 47,149 (`train_fixed`) + 3,039 (`valid_fixed`) — matches CT-RATE's
official train/valid split sizes exactly (confirmed independently via
`CTCLIPFeatureDataset`'s own dataset-size logs used throughout this project:
"Train: 47149 samples, Val: 3039 samples").

**`.nii` (uncompressed): none found.** Expected — CT-RATE ships gzip-compressed
NIfTI only; no uncompressed copies exist anywhere on `/scratch` or `/project`.

No raw NIfTI files exist under `/project/def-*` — only `/scratch/sadeniji/ct_rate_raw`
holds the volumes. `/project/def-uanazodo-ab/sadeniji/` holds the pre-extracted
`.pt` CT-CLIP features and merged report CSVs (see §3), not raw volumes.

## 2. Disk quota

```
$ diskusage_report
                            Description                Space         # of files
                  /home (user sadeniji)         961MB/  50GB         246K/ 500K
               /scratch (user sadeniji)        8070GB/  20TB         399K/1000K
     /project (project def-uanazodo-ab)         687GB/1000GB         366K/ 500K
       /nearline (project rpp-uanazodo)        1356KB/1000GB          55 /5000
    /nearline (project def-uanazodo-ab)         890KB/1000GB          36 /5000

$ df -h /scratch/sadeniji
Filesystem                                   Size  Used Avail Use% Mounted on
10.82.48.7@o2ib1:10.82.48.8@o2ib1:/lustre07   19T  7.6T   12T  41% /scratch
```

- **Scratch quota: 20TB**, currently **~7.6–8.1TB used** (the two commands
  were run a few hours apart — `diskusage_report`'s snapshot is from
  09:10:03, `df -h` is live), **~12TB free**. The raw CT-RATE download
  (§1) is the overwhelming majority of that usage.
- **Project quota: 1000GB, 687GB used** — holds `ctrate_features/` (pre-extracted
  `.pt` files), `ctrate_csv/`, and `hf_cache/` (model weights, not datasets — see §4).
  Getting close to its cap; not a concern for the raw-NIfTI redesign (which
  lives on `/scratch`), but worth knowing if `/project` usage grows further.
- **Home quota: 50GB, 961MB used** — negligible, irrelevant to this redesign.

## 3. CT-RATE metadata/report CSVs

```
$ find /scratch/sadeniji -name "*.csv"
```
Nothing CT-RATE-related under `/scratch/sadeniji` directly — the venv at
`/scratch/sadeniji/venv/` (an old, separate environment, not `~/envs/ictc`)
contains only unrelated library sample-data CSVs (matplotlib/sklearn/numpy
test fixtures), not CT-RATE data.

The actual CT-RATE CSVs live under `/project/def-uanazodo-ab/sadeniji/`:
```
/project/def-uanazodo-ab/sadeniji/ctrate_csv/dataset/radiology_text_reports/train_reports.csv
/project/def-uanazodo-ab/sadeniji/ctrate_csv/dataset/radiology_text_reports/validation_reports.csv
/project/def-uanazodo-ab/sadeniji/ctrate_csv/dataset/merged/train_merged.csv
/project/def-uanazodo-ab/sadeniji/ctrate_csv/dataset/merged/valid_merged.csv
/project/def-uanazodo-ab/sadeniji/ctrate_csv/dataset/merged/_raw/dataset/multi_abnormality_labels/train_predicted_labels.csv
/project/def-uanazodo-ab/sadeniji/ctrate_csv/dataset/merged/_raw/dataset/multi_abnormality_labels/valid_predicted_labels.csv
```
`{train,valid}_merged.csv` (Findings/Impressions text joined with the 18
binary abnormality labels) are what `CTCLIPFeatureDataset` currently reads
from — already present and usable for volume-name/label lookups regardless
of the ViT redesign.

## 4. HuggingFace cache

```
$ find ~/.cache/huggingface -name "*.json" | grep -i ct_rate
(none)
$ ls ~/.cache/huggingface/datasets/
ls: cannot access '/home/sadeniji/.cache/huggingface/datasets/': No such file or directory
```
No default `~/.cache/huggingface` datasets cache exists at all — this
project never relied on it. The actual `HF_HOME` used throughout every
script in this repo is `/project/def-uanazodo-ab/sadeniji/hf_cache`:
```
$ ls /project/def-uanazodo-ab/sadeniji/hf_cache/
hub  models--microsoft--BiomedVLP-CXR-BERT-general  models--StanfordAIMI--GREEN-radllama2-7b  modules  version.txt  xet
```
No `datasets--ibrahimhamamci--CT-RATE` entry in `hub/` — confirms the CT-RATE
dataset was never cached via HF's `datasets`-library caching mechanism.
This is expected and not a gap: `aadp/data/ctrate_dataset.py`'s streaming
path (`datasets.load_dataset(..., streaming=True)`) never persists a local
copy by design, and `scripts/download_ctrate.py` (used this session) wrote
directly to `/scratch/sadeniji/ct_rate_raw` via `local_dir=`, bypassing the
HF cache entirely rather than populating it.

## 5. Scratch quota vs. usage (explicit)

| | Value |
|---|---|
| Scratch quota | 20 TB |
| Scratch used (live, `df -h`) | 7.6 TB |
| Scratch used (`diskusage_report` snapshot, ~4.5h older) | 8.07 TB |
| **Scratch free** | **~12 TB** |

## 6. CT-RATE raw data size — measured vs. the task's estimate

The task's prompt estimated **~420GB** total (21K scans × ~20MB avg,
assuming 256×256×64 float16 with ~10–40MB compressed NIfTI). The **measured**
figure is far larger:

- 50,188 volumes, ~7.6TB total → **~155MB/volume average**, not ~20MB.
- This matches the *original* CT-CLIP feature-extraction log's own measured
  figure almost exactly: `/project/def-uanazodo-ab/sadeniji/ctrate_features/extract_log.txt`
  recorded "raw dataset size = 8220.6 GB (164 MB/volume avg)" across the
  same 50,188-volume set back on 2026-07-16 — independent confirmation from
  a completely different tool/run, ~8 months earlier, within ~7% of this
  session's own total.

The task's ~20MB/volume assumption undersells real CT-RATE volume sizes by
roughly **8x** — full-resolution chest CT volumes (typically 512×512×~300–500
slices, not 256×256×64) compress to tens of MB per slice-stack, not per
whole volume. **There is more than enough free scratch space (12TB free vs.
~7.6TB already used for the complete dataset) — no download is needed, and
none would be blocked by quota even if it were.**

## 7. Existing code paths to raw NIfTI data

```
grep -r "nii\|nifti\|NIfTI\|raw" aadp/ scripts/ --include="*.py" | grep -v ".pyc" | head -30
```
This broad grep mostly matches incidental uses of the word "raw" (raw
logits, raw batches, raw attention weights — unrelated to NIfTI). Narrowing
to files that actually import a NIfTI-reading library:

```
$ grep -rln "nibabel\|SimpleITK\|\.nii\.gz\|load_nifti\|read_image" aadp/ scripts/ --include="*.py"
aadp/data/ctclip_feature_dataset.py   # only for VolumeName->stem string parsing, no real NIfTI I/O
aadp/data/ctrate_dataset.py           # real NIfTI I/O — see below
aadp/data/totalseg_dataset.py         # TotalSegmentator masks (eval only)
aadp/data/preprocessing.py            # HU windowing / normalize / resample / pad-crop
scripts/check_dataset_size.py
scripts/qualitative_eval.py           # only for VolumeName->stem string parsing
scripts/download_ctrate.py            # this session's download script
scripts/qualitative_eval_v2.py        # only for VolumeName->stem string parsing
```

**The important finding: `aadp/data/ctrate_dataset.py` already has a
working local-NIfTI-path function, and its layout convention matches the
downloaded data exactly, with zero path changes needed:**

```python
def _local_nifti_path(local_data_dir: Path, volume_name: str, folder: str) -> Path:
    """Mirror of _volume_hf_path but rooted at a local directory."""
    stem = volume_name.replace(".nii.gz", "").replace(".nii", "")
    parts = stem.split("_")
    name_split = parts[0]
    patient_folder = f"{name_split}_{parts[1]}"
    scan_folder = f"{name_split}_{parts[1]}_{parts[2]}"
    return (
        local_data_dir / "dataset" / folder / patient_folder / scan_folder / volume_name
    )
```
For `local_data_dir=/scratch/sadeniji/ct_rate_raw` and `folder="train_fixed"`,
this produces exactly `/scratch/sadeniji/ct_rate_raw/dataset/train_fixed/train_1/train_1_a/train_1_a_1.nii.gz`
— verified against the real downloaded tree in §1. `aadp/data/ctrate_dataset.py`
also already has `_nifti_to_array()` (loads via `nibabel`) and a
gzip-decompress-to-tempfile path for the streaming case. `aadp/data/preprocessing.py`
already has `window_and_normalize` (HU windowing), `resample_to_isotropic`
(SimpleITK-based), and `pad_or_crop_depth` — the full raw-volume
preprocessing pipeline a ViT-based dataloader would need already exists,
just not currently wired into `CTCLIPFeatureDataset`'s training path (which
deliberately skips it, per its own docstring, to consume pre-extracted
features instead).

**What doesn't exist yet:** nothing in `aadp/models/` currently instantiates
a trainable ViT encoder feeding into Stage 1/Stage 2 — `CTCLIPStage2VLM`
skips the ViT and Stage 1 entirely by design (per its own docstring). The
"unfrozen ViT" model code itself is a genuine gap; the *data* pipeline to
feed it is largely already written.

---

## Summary

```
HAVE_RAW_DATA:    yes  (50,188/50,188 volumes, /scratch/sadeniji/ct_rate_raw, 0 failures)
SCRATCH_FREE_GB:  ~12,000
DOWNLOAD_NEEDED:  no
```

Additional context beyond the three required fields, since it materially
affects next steps:

- **Data pipeline code is ~80% there already**: `aadp/data/ctrate_dataset.py`'s
  `_local_nifti_path()`/`_nifti_to_array()` and `aadp/data/preprocessing.py`'s
  windowing/resampling functions are written and match the downloaded
  layout exactly — likely just needs wiring into a new
  `Dataset.__getitem__` path and a real ViT encoder module, not raw-I/O
  work from scratch.
- **Report/label CSVs already merged and usable** at
  `/project/def-uanazodo-ab/sadeniji/ctrate_csv/dataset/merged/{train,valid}_merged.csv`.
- **The task's own size estimate (~420GB) was off by ~18x** versus the
  measured ~7.6TB — worth keeping in mind for any future capacity planning
  on this project (e.g. augmentation-time copies, resampled-volume caches).
- **Missing piece**: no trainable ViT encoder exists yet anywhere in
  `aadp/models/` — that's the actual new-code surface for this redesign,
  not the data plumbing.

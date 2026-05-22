# Person Re-Identification on PRW

End-to-end **person search** on the [PRW](https://github.com/liangzheng06/PRW-baseline)
dataset (932 identities, 11,816 frames from six cameras, pedestrians annotated with
bounding boxes and identities).

**Task** — given a query crop of a person, find that same person across whole-scene
gallery frames.

**Pipeline** — the problem is split into two independently optimizable stages:

1. **Detection** — locate every person in each gallery frame with an off-the-shelf,
   zero-shot detector (YOLOv12, RF-DETR).
2. **Re-Identification** — encode each detected crop with a **DINOv3 ViT-L/16** backbone
   whose last transformer blocks are fine-tuned with metric learning (ArcFace,
   TripletMargin, NT-Xent) plus a BNNeck projection head, then rank detections by cosine
   similarity to the query embedding.

The main deliverable, `fine_tuned_person_search.ipynb`, runs a 6-way ablation (2 detectors
× 3 losses) and scores each configuration with `eval_search_prw` (mAP and top-1 accuracy).

## Setup

Requires **Python 3.13** and [`uv`](https://docs.astral.sh/uv/).

```bash
# 1. Clone the repository
git clone <REPO-URL>
cd person-re-identification

# 2. Create the virtual environment (uv picks up Python 3.13 from .python-version)
uv venv

# 3. Install the locked dependencies
uv sync
```

The PRW dataset is downloaded and extracted automatically by the first notebook cells, so
no manual dataset step is needed (a Kaggle-hosted archive of ~1 GB).

Launch the notebook with:

```bash
uv run jupyter lab        # then open fine_tuned_person_search.ipynb
```

## Project structure

```
person-re-identification/
├── fine_tuned_person_search.ipynb   # main deliverable: fine-tuned person search + ablation
├── zero_shot_person_search.ipynb    # sibling notebook: pure zero-shot variant
├── solution.ipynb                   # earlier solution notebook
├── generate_query_train.py          # crops labelled training persons into dataset/query_box_train/
├── gallery_caps_B.json              # cached crop captions used by the zero-shot notebook
├── utils/
│   └── eval_function.py             # provided PRW evaluation function (mAP / recall), lightly patched
├── pyproject.toml                   # project metadata and dependencies
├── uv.lock                          # locked dependency versions
├── .python-version                  # pins Python 3.13
│
├── dataset/                         # PRW dataset — auto-downloaded by the notebook
├── detectors/                       # detector weights (yolo12s.pt, rf-detr-base.pth) — auto-downloaded
├── checkpoints/                     # fine-tuned DINOv3 encoder weights (see below)
└── finetune_cache/                  # cached detections & features (see below)
```

## Populating `checkpoints/` and `finetune_cache/`

Running training and feature extraction from scratch is slow (GPU, ~tens of minutes).
To reproduce the notebook results without re-running everything, download the pre-computed
folders and place them at the repository root.

> **Download:** `<INSERT-DOWNLOAD-LINK-HERE>`

The link points to an archive with this layout:

```
checkpoints/        # dinov3_finetune_{arcface,triplet,ntxent}.pt  (+ matching .json)
finetune_cache/     # gallery_dets_*.npy, gallery_vis_*.npy, query_vis_*.npy, results_*.json
```

Extract it so the two folders sit next to the notebook:

```bash
# example — adjust to the archive name/format you publish
unzip person-search-cache.zip -d .
```

Result:

```
person-re-identification/
├── checkpoints/        # ← populated
└── finetune_cache/     # ← populated
```

With both folders in place, run `fine_tuned_person_search.ipynb` top-to-bottom with the
defaults (`PERFORM_TRAINING = False`): it loads the checkpoints and cached features instead
of recomputing them. To retrain or recompute from scratch instead, set `PERFORM_TRAINING = True`.

---

> Replace `<REPO-URL>` and `<INSERT-DOWNLOAD-LINK-HERE>` with the real URLs once available.

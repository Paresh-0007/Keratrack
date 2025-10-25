# Keratrack: Deep Learning–Powered Hair Loss Stage Assessment with Personalized Diet Support

## Abstract

We present Keratrack, an end‑to‑end machine learning system for non‑invasive assessment of male‑pattern hair loss severity from scalp images and delivery of personalized, nutrition‑aware recommendations to support hair health. The system couples a ConvNeXt‑Base image classifier with a production FastAPI backend, a PostgreSQL persistence layer, and a Next.js frontend for interactive user experience. We train on public Roboflow hair‑loss datasets with six severity levels (LEVEL_2–LEVEL_7) and deploy a calibrated softmax classifier that runs on CPU/GPU. On a held‑out set, our model achieves accuracy 85.1%, macro ROC‑AUC 0.962, top‑3 accuracy 98.5%, and low calibration error (ECE ≈ 0.058). We further integrate a lightweight diet AI module that transforms predicted stage and user assessment metadata into individualized nutrient targets, weekly meal plans, and supplement suggestions. We open‑source the complete stack, including training scripts, evaluation artifacts, and an API for reproducible inference.

Index Terms — medical imaging, dermatology, hair loss, computer vision, ConvNeXt, FastAPI, calibration, diet recommendation, MLOps.


## 1. Introduction

Androgenetic alopecia (AGA) affects a significant portion of the adult population and is commonly assessed using scale‑based staging from clinical photographs. Automated, privacy‑preserving, on‑device assessment can increase access and enable longitudinal tracking. Keratrack targets two goals: (i) predict hair loss severity from a single image into six stages (LEVEL_2–LEVEL_7), and (ii) translate the assessment into nutrition guidance tailored to user factors (age, gender, activity, restrictions) to support hair health.

Contributions:
- An end‑to‑end reference system (training → evaluation → deployment) with minimal dependencies.
- A ConvNeXt‑Base classifier with robust preprocessing and K‑Fold training utilities.
- Comprehensive evaluation including accuracy, ROC‑AUC, calibration (ECE), and per‑class metrics.
- A practical API that persists user history and generates personalized diet recommendations from structured inputs.


## 2. System Overview

Keratrack comprises three layers:
- Model: PyTorch + TIMM ConvNeXt‑Base classifier that maps RGB scalp images to six labels: {LEVEL_2,…,LEVEL_7}.
- Backend: FastAPI service with JWT auth, inference endpoint, and diet recommendation endpoints; SQLAlchemy models persisted in PostgreSQL.
- Frontend: Next.js (React) app for login, image upload, result visualization, and diet dashboard.

Code anchors in this repository:
- Model inference: `backend/app/ml_interface.py`
- REST API: `backend/app/main.py`
- ORM schema: `backend/app/models.py`, `backend/app/schemas.py`
- Diet AI: `backend/app/diet_ai.py`
- Training pipeline: `train_ensemble_models.py`
- Evaluation artifacts: `backend/eval_report.json`, `backend/classification_report.txt`
- Frontend pages: `frontend/src/app` (e.g., `predict`, `diet`)


## 3. Data

Datasets: Roboflow “hair-loss” multi‑class classification (CC BY 4.0).
- v2 (June 17, 2024): 1,479 images, preprocessed to 150×150 with rotations.
- v3 (June 19, 2024): 1,475 images, 150×150 + grayscale (CRT phosphor) + rotations.

Train/valid/test CSVs (per Roboflow export) list filenames and one‑hot labels. In our training script we read CSVs and perform Stratified K‑Fold over TRAIN∪VALID by default; TEST is used only for reporting or ensembling. Labels are the six discrete stages: LEVEL_2, LEVEL_3, LEVEL_4, LEVEL_5, LEVEL_6, LEVEL_7 (ordered by severity).

Ethical note: While images are non‑identifying scalp patches, handle any personal data according to CC BY 4.0 and privacy best practices. No sensitive identifiers are used by the model.


## 4. Methods

### 4.1 Preprocessing
- Input: RGB image; convert to RGB if needed; resize to 224×224.
- Normalization: ImageNet mean/std: [0.485, 0.456, 0.406] / [0.229, 0.224, 0.225].
- Inference transform implemented in `ml_interface.py` ensures consistency with training.

Let logits z ∈ R^6, the softmax probabilities are

$$\displaystyle p_i = \frac{e^{z_i}}{\sum_{j=1}^{6} e^{z_j}}.$$

Prediction is $\arg\max_i\ p_i$ with confidence $\max_i p_i$.

### 4.2 Model architecture
- Backbone: ConvNeXt‑Base (timm) with final classification head set to 6 classes.
- Initialization: Pretrained weights for training; for inference, we load a fine‑tuned checkpoint `convnext_base_fold2_best.pth` (place in project root or update path in `ml_interface.py`).
- Loss: Cross‑entropy with label smoothing (0.1) in the training script.
- Optimization: AdamW with OneCycleLR; AMP (autocast) for GPU efficiency.

### 4.3 Training pipeline
Main utility: `train_ensemble_models.py`
- Stratified K‑Fold (default k=5) over TRAIN∪VALID; optional `--holdout_valid` to keep VALID separate.
- Transforms via `get_transforms(IMG_SIZE)` (Albumentations expected).
- Saves per‑fold checkpoints `{model}_fold{f}_best.pth`, OOF/test probabilities for ensembling, and a meta CSV.
- Supports multiple backbones: e.g., `--models convnext_base efficientnet_b3`.

Hyperparameters (typical):
- IMG_SIZE=224, batch size 16, epochs 8, LR 3e‑4, weight decay 1e‑4, label smoothing 0.1.

### 4.4 Diet recommendation module
`backend/app/diet_ai.py` computes individualized nutrient targets from user assessment + hair stage history:
- Maps predicted stages to a numeric severity and progression trend.
- Derives daily targets for key nutrients (protein, iron, biotin, zinc, omega‑3, vitamin D, vitamin C) with stage‑aware multipliers.
- Produces a 7‑day meal plan from templates filtered by dietary restrictions and suggests supplements for higher severity.
- Returns a confidence score based on data sufficiency and assessment completeness.


## 5. Experimental Setup

- Hardware: CPU/GPU (CUDA optional). Training used AMP when CUDA was available.
- Software: Python 3.10+; PyTorch, TIMM, FastAPI, SQLAlchemy. Dependencies listed in `backend/requirements.txt`.
- Data splits: Roboflow exports for TRAIN/VALID/TEST; K‑Fold over TRAIN∪VALID unless `--holdout_valid`.
- Checkpoints: best validation accuracy per fold; an example production checkpoint `convnext_base_fold2_best.pth` is shipped for inference.


## 6. Results

Aggregate metrics from `backend/eval_report.json` and `backend/classification_report.txt`:
- Overall accuracy: 85.1%
- Macro ROC‑AUC: 0.962; Macro Average Precision: 0.904
- Top‑k accuracy: top‑1 85.1%, top‑2 95.5%, top‑3 98.5%
- Calibration (ECE): mean 0.058 (lower is better)
- Per‑class f1 (support in parentheses):
  - LEVEL_2: f1 0.90 (n=15)
  - LEVEL_3: f1 0.82 (n=12)
  - LEVEL_4: f1 0.63 (n=7)
  - LEVEL_5: f1 0.87 (n=16)
  - LEVEL_6: f1 0.90 (n=10)
  - LEVEL_7: f1 1.00 (n=7)

Metric definitions used:
- Accuracy: $\frac{\sum_i \mathbf{1}[\hat{y}_i = y_i]}{N}$
- Macro ROC‑AUC/AP: one‑vs‑rest averaged across classes.
- Log loss: $-\frac{1}{N}\sum_i \log p_{i,y_i}$
- Expected Calibration Error (ECE): $\text{ECE}=\sum_{m=1}^{M}\frac{|B_m|}{N}\,\big|\text{acc}(B_m)-\text{conf}(B_m)\big|$ where bins $B_m$ partition predictions by confidence.

Observations:
- Errors concentrate between adjacent stages (e.g., LEVEL_3 vs LEVEL_4), consistent with ordinal severity.
- Calibration is acceptable for single‑image triage; temperature scaling could further improve ECE if needed.


## 7. System Architecture and API

### 7.1 Data flow
1) User authenticates (JWT). 2) Image upload → FastAPI saves file (`/uploads`) and calls the model. 3) Prediction + confidence saved to `predictions` table. 4) Diet endpoints use the latest stage and the user’s assessment to generate nutrition plans saved in `diet_recommendations`.

### 7.2 Key endpoints (FastAPI)
- POST `/register`, POST `/token` — user creation and login.
- POST `/predict` — multipart image upload; returns `{id, predicted_stage, confidence, created_at}`.
- GET `/history` — latest predictions for the user.
- POST `/diet/assessment` — store user assessment.
- GET `/diet/recommendations` — create or fetch active personalized plan.
- Lifestyle logging: POST `/diet/lifestyle`, GET `/diet/lifestyle/history`, POST `/diet/food-log`.

### 7.3 Database schema (SQLAlchemy)
Entities in `backend/app/models.py`:
- `User`, `Prediction`, `DietAssessment`, `DietRecommendation`, `LifestyleEntry`, `FoodLog`.


## 8. Deployment and Reproducibility

### 8.1 Backend (FastAPI)
Prerequisites:
- PostgreSQL and a connection string set as environment variable `POSTGRES_URL` (e.g., `postgresql+psycopg2://user:password@localhost:5432/keratrack`).
- Python environment with packages from `backend/requirements.txt`.
- Place the trained checkpoint (e.g., `convnext_base_fold2_best.pth`) at repository root or adjust `ckpt_path` in `ml_interface.py`.

Optional commands (Windows PowerShell):
```powershell
# 1) Create and activate a virtual environment
python -m venv venv; .\venv\Scripts\Activate.ps1

# 2) Install backend deps
pip install -r backend/requirements.txt

# 3) Set database URL for this session
$env:POSTGRES_URL = "postgresql+psycopg2://user:password@localhost:5432/keratrack"

# 4) Run API
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 8.2 Frontend (Next.js)
Dependencies in `frontend/package.json` (Next 15, React 19, Tailwind 4). Optional commands:
```powershell
cd frontend
npm install
npm run dev
```
Then navigate to http://localhost:3000. The frontend expects the API at http://localhost:8000 (see fetch calls in `src/app`).

### 8.3 Training
`train_ensemble_models.py` provides a CLI for K‑Fold training and ensembling. Example (GPU recommended):
```powershell
python .\train_ensemble_models.py --models convnext_base --n_splits 5 --epochs 8 --batch_size 16 --num_workers 4
```
You may need to adapt dataset paths inside the script to your local/exported directory layout.


## 9. Limitations and Risks

- Label granularity: The six discrete classes are ordinal, yet trained as nominal; ordinal losses or regression‑to‑rank could reduce adjacent‑class confusion.
- Dataset bias: Roboflow images may not cover all skin tones, lighting, or camera devices; validate on target distribution prior to clinical use.
- Clinical scope: The system is not a medical device; outputs are informational and should not replace professional diagnosis.
- Privacy: Store only what you need. Avoid face images; purge uploaded files or de‑identify paths as policy.


## 10. Future Work

- Ordinal classification or distributional regression for stage estimation.
- Uncertainty quantification and temperature scaling for improved calibration.
- Semi‑supervised/active learning to expand coverage to new populations and devices.
- Richer diet engine: larger food databases, macro‑to‑recipe optimization, and adherence modeling.
- On‑device inference (Core ML / NNAPI) for privacy and latency.


## 11. Related Work (brief)

- ConvNeXt: Liu et al., “A ConvNet for the 2020s,” CVPR 2022.
- Assessing AGA via computer vision: prior works typically rely on classical features or limited CNN baselines; strong modern backbones improve robustness under real‑world capture conditions.


## 12. References

1) Z. Liu et al., “A ConvNet for the 2020s,” 2022 IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR), pp. 11976–11986.
2) Wightman, R., TIMM: PyTorch Image Models, https://github.com/huggingface/pytorch-image-models
3) Paszke, A. et al., PyTorch: An Imperative Style, High‑Performance Deep Learning Library, NeurIPS 2019.
4) FastAPI, https://fastapi.tiangolo.com
5) Roboflow Datasets, https://universe.roboflow.com


## 13. Appendix

### A. File map (selected)
- `backend/app/ml_interface.py` — model loading and preprocessing; `LABELS=['LEVEL_2',…,'LEVEL_7']`.
- `backend/app/main.py` — API endpoints for auth, prediction, history, and diet.
- `backend/app/models.py` — SQLAlchemy ORM tables.
- `backend/app/schemas.py` — Pydantic models for request/response.
- `backend/app/diet_ai.py` — deterministic recommendation engine.
- `backend/requirements.txt` — Python dependencies.
- `frontend/src/app` — Next.js pages (`predict`, `diet`, `login`, etc.).

### B. Reuse checklist (for IEEE artifact review)
- Code and trained weights path documented.
- Data sources and licenses enumerated; no restricted content included.
- Deterministic inference provided; training seed initialization in script.
- Metrics and evaluation scripts provided; JSON and text reports included in `backend/`.


## License and Attribution

- Dataset: CC BY 4.0 (as listed in Roboflow READMEs in `backend/hair-loss.*`).
- Code: add appropriate license in repository root if publishing. Please ensure any additional dependencies comply with intended distribution.


## How to Cite

If you use Keratrack in academic work, please cite this repository and the ConvNeXt paper. An example BibTeX (to customize):

```
@software{keratrack2025,
  author = {Gupta, Paresh and contributors},
  title = {Keratrack: Hair Loss Stage Assessment with Personalized Diet Support},
  year = {2025},
  url = {https://github.com/Paresh-0007/Keratrack}
}
```
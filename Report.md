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

### 4.5 Diet recommendation flow and formulas (detailed)

This section formalizes the deterministic pipeline implemented in `backend/app/diet_ai.py`.

1) Hair stage time‑series → severity and trend
- Stage to severity mapping: LEVEL_2..LEVEL_7 → s ∈ [0.2, 1.0] (see HAIR_STAGE_SEVERITY).
- Let the most recent prediction be at time t0 and the oldest at tn. Define progression rate per month:

  $\displaystyle r = \frac{s(t_0) - s(t_n)}{\max(1,\ \text{days}(t_0,t_n))}\times 30.$

- Trend label: worsening if r > 0.05; improving if r < −0.05; else stable.

2) Personalized nutrient targets
- Start from nutrient‑specific base needs b_n and hair‑loss multipliers m_n (domain‑informed constants in HAIR_NUTRIENTS).
- Adjust for gender where applicable (e.g., iron 18 mg women, 8 mg men; omega‑3 1.1 g women, 1.6 g men; vitamin C 75 mg women, 90 mg men).
- Protein scales with weight W (kg): $b_{protein} = 0.8\,\text{g/kg}\times W$.
- Hair severity multiplier: $h = 1 + s$ where s is current severity.
- Urgency multiplier for worsening trend: $u = 1.2$ if r > 0 else $u = 1.0$.
- Final daily target for nutrient n:

  $\displaystyle T_n = b_n \times m_n \times h \times u.$

Units are per nutrient: protein (g), iron (mg), biotin (mcg), zinc (mg), omega‑3 (g), vitamin D (IU), vitamin C (mg).

3) Meal plan selection
- For each meal candidate with nutrient profile A_n, target split assumes 3 main meals:

  $\displaystyle T^{(meal)}_n = T_n/3.$

- Scoring a meal:
  - If $A_n \ge 0.8\,T^{(meal)}_n$ add 1 point; else if $A_n \ge 0.5\,T^{(meal)}_n$ add 0.5 point.
  - Sum over nutrients; pick argmax per meal slot (breakfast/lunch/dinner) after filtering by dietary restrictions.

4) Snack suggestions
- Heuristic short‑list emphasizing omega‑3, protein, iron/zinc, vitamin C.

5) Supplement heuristics
- If severity s > 0.6: add Biotin Complex (≈5,000 mcg/d), Iron + Vitamin C (≈25 mg + 100 mg), Omega‑3 fish oil (≈1,000 mg EPA+DHA/d), timing as specified.
- If male or s > 0.7: add Zinc (≈15 mg/d).
- Note: Doses are illustrative defaults; real use should be clinician‑guided.

6) Confidence score
- Base confidence 0.7; data sufficiency $c_d = \min(1, N/6)$ from last N predictions; assessment completeness $c_a = K/10$ where K is count of non‑null fields.

  $\displaystyle C = \min\big(0.7 + 0.2\,c_d + 0.1\,c_a,\ 0.95\big).$

7) Output bundle
- Returns: nutrient targets {T_n, unit, sources, importance}, 7‑day meal plan, supplement list, reasoning string, hair analysis {s, r, trend, confidence}, next review date = now + 30 days.

Design notes and alignment with literature:
- The mapping from hair loss stage to target nutrient intensity reflects clinical evidence that protein adequacy, iron status, zinc, certain B‑vitamins (biotin), omega‑3 fatty acids, and vitamin D/C impact hair health [R1–R3].
- The recommendation engine is knowledge‑based and rule‑driven (not a collaborative filter). Similar approaches are common in nutrition recommender systems that encode expert rules and user constraints rather than learning preferences from large user histories [R4–R5].

### 4.5.1 Flowchart (engine overview)

```mermaid
flowchart TD
    A["User Assessment
    (age, gender, weight, height,
    activity, restrictions, conditions)"]
    B["Prediction History
    (stage, confidence, timestamp)"]
    C["Hair Analysis
    Compute severity s and trend r"]
    D["Base Needs b_n
    (gender-, weight-specific)"]
    E["Multipliers
    Hair m_n, severity h=1+s, urgency u"]
    F["Daily Targets
    T_n = b_n * m_n * h * u"]
    G["Meal Library
    (nutrient profiles, constraints)"]
    H["Filter by Restrictions"]
    I["Per-meal Scoring
    against T_n/3"]
    J["7-day Meal Plan"]
    K["Supplement Heuristics"]
    L["Confidence C"]
    M["Final Bundle
    {nutrients, meal plan, supplements,
    reasoning, hair analysis, C}"]

    A --> C
    B --> C
    C --> E
    A --> D
    D --> F
    E --> F
    F --> I
    G --> H --> I --> J --> M
    F --> K --> M
    C --> L --> M

```

### 4.6 Formulas used (with definitions)

Let the set of tracked nutrients be N = {protein, iron, biotin, zinc, omega3, vitamin_d, vitamin_c}.

Variables and constants:
- s ∈ [0,1]: current hair loss severity (from stage mapping, e.g., LEVEL_2→0.2, …, LEVEL_7→1.0)
- r: progression rate per month derived from history (see below)
- W (kg): user body weight
- sex ∈ {male, female}
- b_n: base daily need for nutrient n
  - protein: 0.8 g/kg × W
  - iron: 18 mg (female), 8 mg (male)
  - omega3: 1.1 g (female), 1.6 g (male)
  - vitamin_c: 75 mg (female), 90 mg (male)
  - biotin: 30 mcg; zinc: 8 mg (female)/11 mg (male); vitamin_d: 600 IU
- m_n ≥ 1: hair‑loss multiplier per nutrient (domain constants in code)
- h = 1 + s: severity multiplier
- u = 1.2 if r > 0 else 1.0: urgency multiplier for worsening trend

Progression rate:

$$ r = \frac{s(t_0) - s(t_n)}{\max(1,\ \text{days}(t_0,t_n))}\times 30. $$

Trend rule:

$$\text{trend} = \begin{cases}
  ext{worsening} & r > 0.05,\\
  ext{improving} & r < -0.05,\\
  ext{stable} & \text{otherwise.}
\end{cases}$$

Daily nutrient target for each n ∈ N:

$$ T_n = b_n\times m_n\times h\times u. $$

Per‑meal target (3 main meals):

$$ T^{(meal)}_n = T_n / 3. $$

Meal scoring (for a candidate meal with nutrient amounts A_n):

$$\text{score} = \sum_{n\in N} \Big( \mathbf{1}[A_n \ge 0.8\,T^{(meal)}_n] + 0.5\,\mathbf{1}[0.5\,T^{(meal)}_n \le A_n < 0.8\,T^{(meal)}_n] \Big). $$

Supplements (heuristic):
- if s > 0.6 → add biotin complex (~5000 mcg/d), iron+vitamin C (~25 mg + 100 mg), omega‑3 (~1000 mg EPA/DHA/d)
- if sex=male or s > 0.7 → add zinc (~15 mg/d)

Confidence score:

Let Np be number of recent predictions used (bounded by 6), and K be count of non‑null assessment fields (out of 10):

$$ c_d = \min(1, N_p/6),\quad c_a = K/10, \quad C = \min(0.7 + 0.2\,c_d + 0.1\,c_a,\ 0.95). $$

Boundary conditions and safeguards:
- Clamp s to [0,1]. If history < 2 samples, set r=0 and trend=insufficient_data.
- Enforce non‑negative T_n; cap to safe upper bounds if integrating RDA/UL knowledge.
- Respect dietary restrictions by filtering meals before scoring.


### 4.7 Worked example: from inputs to diet plan

Example user assessment and history:
- Age 30, gender female, weight W=65 kg, height 165 cm, activity moderate
- Dietary restrictions: ["vegetarian"]; health conditions: []; medications: []
- Prediction history (last 90 days): tn: LEVEL_4 (s=0.6), t0: LEVEL_5 (s=0.8)

Step 1 — Severity and trend
- Current severity s = 0.8 (LEVEL_5)
- Progression rate per month: r = (0.8 − 0.6)/90 × 30 = 0.0667 → r > 0.05 ⇒ trend = worsening

Step 2 — Multipliers
- Severity multiplier h = 1 + s = 1.8
- Urgency multiplier u = 1.2 (since r > 0)
- Combined multiplier h×u = 2.16

Step 3 — Base needs b_n and hair multipliers m_n
- Protein b = 0.8 g/kg × 65 = 52.0 g; m = 1.3
- Iron b = 18 mg (female); m = 1.5
- Biotin b = 30 mcg; m = 100
- Zinc b = 8 mg (female); m = 1.5
- Omega‑3 b = 1.1 g (female); m = 2.0
- Vitamin D b = 600 IU; m = 2.0
- Vitamin C b = 75 mg (female); m = 1.2

Step 4 — Daily targets via Tn = bn × mn × h × u
- Protein: 52.0 × 1.3 × 2.16 = 146.02 g/day
- Iron: 18 × 1.5 × 2.16 = 58.32 mg/day
- Biotin: 30 × 100 × 2.16 = 6,480 mcg/day
- Zinc: 8 × 1.5 × 2.16 = 25.92 mg/day
- Omega‑3: 1.1 × 2.0 × 2.16 = 4.752 g/day
- Vitamin D: 600 × 2.0 × 2.16 = 2,592 IU/day
- Vitamin C: 75 × 1.2 × 2.16 = 194.40 mg/day

Per‑meal targets (divide by 3):
- Protein 48.67 g; Iron 19.44 mg; Biotin 2,160 mcg; Zinc 8.64 mg; Omega‑3 1.584 g; Vitamin D 864 IU; Vitamin C 64.80 mg

Step 5 — Filter meal library by restrictions
- Remove non‑vegetarian meals (e.g., salmon dinner). Remaining examples:
  - high_protein_breakfast: {protein 25, iron 3, biotin 25, omega‑3 0.3}
  - iron_rich_lunch: {protein 18, iron 6, zinc 3, vitamin C 80}

Step 6 — Score meals vs per‑meal targets
- Breakfast candidates (id contains "breakfast"): high_protein_breakfast only
  - Protein 25 vs 48.67 → ≥ 0.5×T (≥24.34) but < 0.8×T (38.94) ⇒ +0.5
  - Iron 3 vs 19.44 ⇒ +0
  - Biotin 25 vs 2,160 ⇒ +0
  - Omega‑3 0.3 vs 1.584 ⇒ +0
  - Total breakfast score = 0.5 → selected
- Lunch candidates (id contains "lunch"): iron_rich_lunch only
  - Protein 18 vs 48.67 ⇒ +0
  - Iron 6 vs 19.44 ⇒ +0
  - Zinc 3 vs 8.64 ⇒ +0
  - Vitamin C 80 vs 64.80 ⇒ ≥ 0.8×T (51.84) ⇒ +1.0
  - Total lunch score = 1.0 → selected
- Dinner candidates (id contains "dinner"): none remaining after filter → fallback to first remaining meal by engine logic (high_protein_breakfast)

Step 7 — Snack suggestions (top 3)
- Handful of walnuts (omega‑3)
- Greek yogurt with berries (protein, biotin)
- Pumpkin seeds (zinc, iron)

Step 8 — Supplements (heuristic)
- s > 0.6 ⇒ add Biotin Complex (~5000 mcg/d), Iron + Vitamin C (~25 mg + 100 mg), Omega‑3 (~1000 mg EPA/DHA/d)
- s > 0.7 ⇒ add Zinc (~15 mg/d)

Step 9 — Confidence
- Assume Np = 2 predictions available → cd = min(1, 2/6) = 0.33
- Assume K = 9 of 10 assessment fields filled → ca = 0.9
- C = min(0.7 + 0.2×0.33 + 0.1×0.9, 0.95) ≈ 0.86

Resulting Day‑1 plan (engine preview):
- Breakfast: High‑Protein Breakfast (score 0.5)
- Lunch: Iron‑Rich Lunch (score 1.0)
- Dinner: High‑Protein Breakfast (fallback due to restriction)
- Snacks: walnuts; Greek yogurt with berries; pumpkin seeds
- Daily targets: as computed above (Tn)
- Supplements: biotin complex, iron+vitamin C, omega‑3, zinc
- Confidence: ~0.86

Notes:
- This is a knowledge‑based demonstration; for clinical use, integrate RDAs/ULs, physician guidance, and a richer vegetarian meal library to hit protein/omega‑3 targets without fish (e.g., tofu/tempeh, legumes, chia/flax, algae‑based DHA).

#### 4.7.1 Numeric summary table (Example A: female, vegetarian)

| Nutrient   | b_n (base) | m_n | h | u | T_n (daily target) | T_n/3 (per meal) |
|------------|------------|-----|---|---|---------------------|------------------|
| Protein    | 52.00 g    | 1.3 |1.8|1.2| 146.02 g           | 48.67 g          |
| Iron       | 18.00 mg   | 1.5 |1.8|1.2| 58.32 mg           | 19.44 mg         |
| Biotin     | 30.00 mcg  | 100 |1.8|1.2| 6,480 mcg          | 2,160 mcg        |
| Zinc       | 8.00 mg    | 1.5 |1.8|1.2| 25.92 mg           | 8.64 mg          |
| Omega‑3    | 1.10 g     | 2.0 |1.8|1.2| 4.752 g            | 1.584 g          |
| Vitamin D  | 600 IU     | 2.0 |1.8|1.2| 2,592 IU           | 864 IU           |
| Vitamin C  | 75.00 mg   | 1.2 |1.8|1.2| 194.40 mg          | 64.80 mg         |

Values computed via T_n = b_n × m_n × h × u with h = 1 + s and u based on trend.

#### 4.7.2 Second example (Example B: male, stable trend)

Inputs:
- Age 35, gender male, weight W=75 kg, height 176 cm, activity moderate
- Restrictions: none
- History (60 days): tn: s=0.42, t0: s=0.40 ⇒ r = (0.40 − 0.42)/60 × 30 = −0.01 ⇒ |r| < 0.05 ⇒ trend = stable

Multipliers:
- s = 0.40 ⇒ h = 1 + s = 1.4; u = 1.0 (not worsening)

Base needs (male) and hair multipliers:
- Protein b = 0.8 g/kg × 75 = 60.0 g; m = 1.3 ⇒ T = 60.0 × 1.3 × 1.4 × 1.0 = 109.20 g
- Iron b = 8 mg; m = 1.5 ⇒ 16.80 mg
- Biotin b = 30 mcg; m = 100 ⇒ 4,200 mcg
- Zinc b = 11 mg; m = 1.5 ⇒ 23.10 mg
- Omega‑3 b = 1.6 g; m = 2.0 ⇒ 4.48 g
- Vitamin D b = 600 IU; m = 2.0 ⇒ 1,680 IU
- Vitamin C b = 90 mg; m = 1.2 ⇒ 151.20 mg

Per‑meal targets (÷3): Protein 36.40 g; Iron 5.60 mg; Biotin 1,400 mcg; Zinc 7.70 mg; Omega‑3 1.49 g; Vitamin D 560 IU; Vitamin C 50.40 mg.

Supplements (per heuristics):
- Male ⇒ add Zinc Picolinate 15 mg/d even though severity ≤ 0.6.
- No high‑severity bundle (biotin complex, iron+vitC, omega‑3) because s ≤ 0.6.

Confidence example:
- Assume Np = 4 ⇒ c_d = min(1, 4/6) ≈ 0.67; K = 10 ⇒ c_a = 1.0
- C = min(0.7 + 0.2×0.67 + 0.1×1.0, 0.95) ≈ 0.93

Remarks:
- Compared to Example A (worsening, s=0.8), lower s and u=1.0 reduce targets substantially.
- Non‑restricted meals (e.g., salmon dinner) would likely score higher against omega‑3/Vit D targets; vegetarian or vegan constraints should pivot to algae‑DHA, flax/chia, fortified foods.


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

R1) Almohanna MN, Ahmed AA, Tsatalis JP, Tosti A. “The Role of Vitamins and Minerals in Hair Loss: A Review.” Dermatology and Therapy (Heidelb). 2019.

R2) Trost LB, Bergfeld WF, Calogeras E. “The diagnosis and treatment of iron deficiency and its potential relationship to hair loss.” Journal of the American Academy of Dermatology. 2006.

R3) Rushton DH, Norris MJ, Dover R, Busuttil N. “Causes of hair loss and the developments in hair rejuvenation.” International Journal of Cosmetic Science. 2002. (See also Rushton DH. “Nutritional factors and hair loss.” Clinical and Experimental Dermatology. 2002.)

R4) Trattner C, Elsweiler D. “Food Recommender Systems.” In: Social Information Access. Springer, 2018. (Survey of knowledge‑based and health‑aware food recommenders.)

R5) Elsweiler D, Trattner C, Harvey M. “Exploiting food choice biases for healthier recipe recommendation.” ACM SIGIR. 2015. (Example of health‑oriented recommendation constraints.)

R6) Burke R. “Knowledge‑based recommender systems.” Encyclopedia of Library and Information Systems. 2000. (Foundational overview of rule/constraint‑driven recommenders.)

R7) Freyne J, Berkovsky S. “Intelligent Food Planning: Personalized Recipe Recommendation.” ACM UMAP. 2010. (Uses constraints/objectives akin to formula‑based matching.)

R8) Ge M., Elahi M., Fernández-Tobías I., Ricci F., Massimo D. “Health-aware Food Recommender System.” Workshop on Food Recommender Systems, RecSys. 2015. (Health constraints and rule‑based components.)

R9) Zeevi D. et al. “Personalized nutrition by prediction of glycemic responses.” Cell. 2015. (Algorithmic personalized nutrition; demonstrates data‑driven formulation of diet recommendations.)


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
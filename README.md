# ISL-Confidence: Single-Cell Confidence & Interpretability for In-Silico Labeling

<img src="images/overview.png" alt="Project overview" width="520"/>

[Paper / preprint](#)

---

## Project Description

This repository hosts two tightly related but **separable** projects:

- **`single_cell/`** — Single-cell confidence model that predicts per-cell error/quality for ISL outputs using 3D patches.  
- **`interpretability/`** — Patch level confidence for ISL with detailed analysis pipelines.

Each folder has **its own Python environment** and **paths/configs**. Keep them isolated.

### Environments & Paths (per folder)

- A `requirements.txt` resides **inside each project folder**.
- Recommended: Isolated venv per folder:
  ```bash
  cd single_cell
  python -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  ```
  (repeat for `interpretability/`)
- Paths are configured per project. TODO: explain this.

### Brief research & code overview

- **Goal**: Quantify **uncertainty at single-cell resolution** for ISL predictions and provide **faithful explanations** for why predictions succeed/fail.  
- **Significance**: Improves trust, troubleshooting, and deployment of ISL in biological imaging by telling **where** and **why** a prediction is reliable.  
- **Key features**
  - *3D confidence model* that ingests ISL predictions + explanation masks and regresses a per-cell/per-patch quality/error target (e.g., PCC-derived).
  - *Detailed analysis* to understand cell-level or FOV-level mistakes.
  - *Useful applications* that can be performed using this method.
---

## Data (TODO)

- **`single_cell/` dataset**  
  - 3D **z-stacks**, per-cell patches (e.g., 128×128 XY), targets derived from correlation/quality metrics.  
  - Typical sample contains: prediction volume, (optional) importance mask volume, and per-cell target(s).  
  - Folder structure and preprocessing scripts described in `single_cell/README.md`.

- **`interpretability/` dataset**  
  - Paired unlabeled→labeled microscopy volumes for ISL (e.g., brightfield → fluorescence), plus optional context.  
  - Trained to produce **importance masks** explaining ISL predictions; supports 2D/3D.  
  - Data layout, normalization, and configs in `interpretability/README.md`.

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/GadMil/Interpretability
cd Interpretability

# --- Single-cell model ---
cd single_cell
conda create -n single_cell python=3.9.15
conda activate single_cell
pip install -r requirements_windows.txt

# --- Confidence model ---
cd ../interpretability
conda create -n confidence python=3.10.14
conda activate confidence
pip install -r requirements_windows.txt
```

---

## Citation & Credit (compact) TODO

If you use this **code** or **data**, please **cite** the associated paper and this repository.

**BibTeX (repo):**
```bibtex
@misc{isl_confidence_repo,
  title        = {ISL-Confidence: Single-Cell Confidence & Interpretability for In-Silico Labeling},
  author       = {Your Name and Collaborators},
  year         = {2025},
  howpublished = {\url{https://github.com/<org>/<repo>}}
}
```

**BibTeX (paper/preprint placeholder):**
```bibtex
@article{isl_confidence_paper,
  title   = {Quantifying Uncertainty in In-Silico Labeling via Single-Cell Confidence and Mask-Based Interpretability},
  author  = {Your Name and Collaborators},
  journal = {Preprint},
  year    = {2025}
}
```

Credits and dataset attributions are listed in each sub-README.

---

## Repository Layout

```
.
├─ README.md                # (this file)
├─ README_Figures/                  # figures used by README (e.g., overview.png)
├─ single_cell/             # single-cell confidence model
└─ interpretability/        # mask-based interpretability
```

---

**License & Contact**  
See `LICENSE` in main folder. For questions or collaboration, open an issue or contact the maintainer listed in the sub-READMEs.

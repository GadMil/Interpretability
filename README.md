# ISL-Confidence: Single-Cell Confidence & Interpretability for In-Silico Labeling

> Minimal, repo-level README. Each subproject has its own full README.

<img src="images/overview.png" alt="Project overview" width="520"/>

[Paper / preprint](#) · [Single-Cell README](single_cell/README.md) · [Interpretability README](interpretability/README.md)

---

## Project Description

This repository hosts two tightly related but **separable** projects:

- **`single_cell/`** — Single-cell **confidence model** that predicts per-cell error/quality for ISL outputs using 3D patches.  
- **`interpretability/`** — **Mask-based interpretability** for ISL (importance masks that explain what image regions drive predictions).

Each folder has **its own Python environment** and **paths/configs**. Keep them isolated.

### Environments & Paths (per folder)

- A `requirements.txt` resides **inside each project folder**.
- Recommended: Python ≥ 3.10, isolated venv per folder:
  ```bash
  cd single_cell
  python -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  ```
  (repeat for `interpretability/`)
- Paths are configured per project (e.g., `configs/*.yaml` or environment variables like `DATA_ROOT`, `RUNS_DIR`). See the sub-READMEs for exact keys and examples.

### Brief research & code overview

- **Goal**: Quantify **uncertainty at single-cell resolution** for ISL predictions and provide **faithful explanations** for why predictions succeed/fail.  
- **Significance**: Improves trust, troubleshooting, and deployment of ISL in biological imaging by telling **where** and **why** a prediction is reliable.  
- **Key features**
  - *Single-Cell*: 3D patch-based model (e.g., modified ResNet-18) that ingests ISL predictions (+ optional explanation masks) and regresses a per-cell quality/error target (e.g., PCC-derived).
  - *Interpretability*: Mask Interpreter that learns **importance masks** via noise-based objectives; supports 3D and context-aware ISL backbones.
  - *Modular*: Independent training/eval loops, configs, logging, and visualization utilities in each folder.

---

## Data (high-level)

*(Full details in each sub-README.)*

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
# Single-cell confidence model
cd single_cell
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train.py --config configs/base.yaml

# Interpretability (Mask Interpreter)
cd ../interpretability
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train.py --config configs/base.yaml
```

---

## Citation & Credit (compact)

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
├─ images/                  # figures used by README (e.g., overview.png)
├─ single_cell/             # single-cell confidence model (+ its own README)
└─ interpretability/        # mask-based interpretability (+ its own README)
```

---

**License & Contact**  
See `LICENSE` (if included). For questions or collaboration, open an issue or contact the maintainer listed in the sub-READMEs.

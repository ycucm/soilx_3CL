# SoilX 3CL

A contrastive learning based algorithm for predicting soil chemical concentrations (M, N, P, K, C, Al)  
from dielectric permittivity and VNIR spectral features.

---

## Dataset Format

| Column | Description |
|:-------|:-------------|
| `id` | Sample ID (`#M1`, `#N2`, `#0` for reference) |
| `M`, `N`, `P`, `K`, `C`, `Al` | Ground-truth chemical concentrations |
| `dielectric_permittivity` | dielectric constant |
| `vnir_0 ... vnir_1650` | VNIR reflectance/intensity features |

---

## Train

```bash
python train_v4.py \
  --csv dataset/training.csv \
  --save_dir runs/soilx \
  --epochs 5000
```
Saves model:

runs/soilx/soilx_pretrained_YYYYMMDD_HHMMSS.pth


## Inference
```
python inference_v4.py \
  --csv dataset/test_lab.csv \
  --model runs/soilx/<the model file name above> \
  --save_csv runs/soilx/preds_test_lab_v3.csv
```
Outputs predicted columns:
pred_M, pred_N, pred_P, pred_K, pred_C, pred_Al

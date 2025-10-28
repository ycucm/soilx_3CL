# SoilX 3CL

##  Overview
A contrastive learning based algorithm for predicting soil chemical concentrations (M, N, P, K, C, Al)  
from dielectric permittivity and VNIR spectral features.

---

## Environment

```bash
git clone https://github.com/ycucm/soilx_3CL.git
cd soilx_3CL

conda create -n soilx3cl python=3.9 -y
conda activate soilx3cl

# (Optional) Install PyTorch with CUDA 12.1 wheels. Use cu118 if your CUDA is 11.8.

pip install -r requirements.txt
```

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
python train.py \
  --csv dataset/training.csv \
  --save_dir runs/soilx \
  --epochs 5000
```
Saves model:

runs/soilx/soilx_pretrained_YYYYMMDD_HHMMSS.pth


## Inference
```
python inference.py \
  --csv dataset/test_lab.csv \
  --model runs/soilx/<the model file name above> \
  --save_csv runs/soilx/<output csv file name>
```
Outputs predicted columns:
pred_M, pred_N, pred_P, pred_K, pred_C, pred_Al

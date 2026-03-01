# GeoAI Raster Classifier (Sentinel-2 RGB)

This project builds a **Decision Tree** machine learning model to classify a GeoTIFF raster using **3 bands only (RGB)** into 3 classes:
- **1 = Water**
- **2 = Agricultural areas**
- **3 = Urban areas**

The model is trained offline and then used inside a **Streamlit** web app to classify new rasters and download the result as GeoTIFF.

---

## Data
- Satellite: **Sentinel-2 L2A**
- Bands used (RGB): **3 bands only**
- Ground truth: polygons created in QGIS with attribute `class_id` (1/2/3)

> Note: The `data/` folder is not included in this repository (large files).

---

## Files
- `training_script.py` : Offline training script (read raster, rasterize labels, reshape, split, train, evaluate, save model)
- `app.py` : Streamlit application (upload GeoTIFF, choose RGB bands, classify, download result)
- `model.pkl` : Trained model bundle
- `requirements.txt` : App dependencies
- `metrics.json` : Train/Validation/Test metrics and confusion matrices

---

## How to run (local)

### 1) Install dependencies
```bash
pip install -r requirements.txt

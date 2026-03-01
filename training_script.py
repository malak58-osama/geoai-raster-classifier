import json
import numpy as np
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# ----------------- إعدادات المشروع -----------------
RASTER_PATH = "data/raster_rgb.tif"
LABELS_PATH = "data/labels.shp"
LABEL_FIELD = "class_id"

# نحن ندرّب على 3 باندات فقط (RGB) من الراستر متعدد الباندات
# في مشروعك: Band1=R, Band2=G, Band3=B (حسب دمجك في QGIS)
BANDS_1BASED = [1, 2, 3]

RANDOM_STATE = 42
MAX_DEPTH = 12

MODEL_OUT = "model.pkl"
METRICS_OUT = "metrics.json"

LABELS_ORDER = [1, 2, 3]
CLASS_NAMES = {1: "water", 2: "agriculture", 3: "urban"}


def evaluate_split(split_name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS_ORDER)
    return {
        "split": split_name,
        "confusion_matrix(labels=[1,2,3])": cm.tolist(),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def main():
    # 1) قراءة الراستر
    with rasterio.open(RASTER_PATH) as src:
        raster_all = src.read()  # (bands,H,W)
        meta = src.meta.copy()
        transform = src.transform
        raster_crs = src.crs
        nodata = src.nodata
        H, W = src.height, src.width
        count = src.count

    print("Raster loaded")
    print("Bands:", count, "| Shape:", raster_all.shape, "| CRS:", raster_crs)

    # 2) اختيار 3 باندات فقط
    idx = [b - 1 for b in BANDS_1BASED]
    if max(idx) >= raster_all.shape[0]:
        raise ValueError("الباندات المختارة غير موجودة في الراستر.")
    raster = raster_all[idx, :, :].astype(np.float32)  # (3,H,W)

    # 3) قراءة الشيب فايل
    gdf = gpd.read_file(LABELS_PATH)
    print("Labels loaded | polygons:", len(gdf), "| CRS:", gdf.crs)

    # 4) توحيد CRS إذا لزم
    if gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)
        print("Reprojected labels to:", raster_crs)

    # 5) Rasterize للـ labels على نفس Grid الراستر
    shapes = [(geom, int(val)) for geom, val in zip(gdf.geometry, gdf[LABEL_FIELD])]
    y_raster = rasterize(
        shapes=shapes,
        out_shape=(H, W),
        transform=transform,
        fill=0,        # 0 = بدون label
        dtype="uint8",
    )

    # 6) تحويل البيانات من (H,W,3) إلى جدول (N,3) + y
    # X لكل البكسلات ثم نفلتر فقط المعلّمة
    X_all = np.transpose(raster, (1, 2, 0)).reshape(-1, 3)  # (H*W,3)
    y_all = y_raster.reshape(-1)                             # (H*W,)

    # mask للبكسلات المعلّمة + الصالحة
    mask = (y_all > 0)
    mask &= np.all(np.isfinite(X_all), axis=1)
    if nodata is not None:
        mask &= np.all(X_all != nodata, axis=1)

    X = X_all[mask].astype(np.float32)
    y = y_all[mask].astype(np.uint8)

    print("Total labeled pixels:", len(y))
    unique, counts = np.unique(y, return_counts=True)
    print("Class pixel counts:", dict(zip(unique.tolist(), counts.tolist())))

    # 7) تقسيم Train / Validation / Test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
    )
    print("Split sizes:", {"train": len(y_train), "val": len(y_val), "test": len(y_test)})

    # 8) تدريب Decision Tree
    model = DecisionTreeClassifier(
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    # 9) تقييم منفصل Train/Val/Test
    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)

    results = [
        evaluate_split("train", y_train, pred_train),
        evaluate_split("validation", y_val, pred_val),
        evaluate_split("test", y_test, pred_test),
    ]

    metrics = {
        "raster_path": RASTER_PATH,
        "labels_path": LABELS_PATH,
        "bands_used_1based": BANDS_1BASED,
        "class_names": CLASS_NAMES,
        "model": "DecisionTreeClassifier",
        "model_params": {
            "max_depth": MAX_DEPTH,
            "random_state": RANDOM_STATE,
            "class_weight": "balanced",
        },
        "n_samples": {
            "train": int(len(y_train)),
            "validation": int(len(y_val)),
            "test": int(len(y_test)),
        },
        "results": results,
    }

    with open(METRICS_OUT, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 10) حفظ النموذج
    bundle = {
        "model": model,
        "bands_used_1based": BANDS_1BASED,
        "class_names": CLASS_NAMES,
        "labels_order": LABELS_ORDER,
    }
    joblib.dump(bundle, MODEL_OUT, compress=3)

    print("Saved:", MODEL_OUT)
    print("Saved:", METRICS_OUT)

    # طباعة سريعة
    for r in results:
        print("\n---", r["split"], "---")
        print("Accuracy:", r["accuracy"])
        print("Precision_macro:", r["precision_macro"])
        print("Recall_macro:", r["recall_macro"])
        print("F1_macro:", r["f1_macro"])
        print("Confusion matrix [1,2,3]:")
        for row in r["confusion_matrix(labels=[1,2,3])"]:
            print(row)


if __name__ == "__main__":
    main()
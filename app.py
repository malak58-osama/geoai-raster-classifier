import numpy as np
import streamlit as st
import rasterio
from rasterio.io import MemoryFile
import joblib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

MODEL_PATH = "model.pkl"

@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)

def read_geotiff(uploaded_file):
    with rasterio.open(uploaded_file) as src:
        arr = src.read()  # (bands,H,W)
        meta = src.meta.copy()
        nodata = src.nodata
    return arr, meta, nodata

def make_rgb_preview(arr, r, g, b):
    rgb = arr[[r-1, g-1, b-1], :, :].astype(np.float32)  # (3,H,W)
    rgb = np.transpose(rgb, (1, 2, 0))  # (H,W,3)

    vals = rgb[np.isfinite(rgb)]
    if vals.size > 0:
        p2, p98 = np.percentile(vals, (2, 98))
        rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0, 1)
    else:
        rgb = np.zeros_like(rgb)

    return rgb

def classify_pixels(arr, nodata, model, r, g, b):
    if max(r, g, b) > arr.shape[0]:
        raise ValueError("اختيار الباندات غير صالح: عدد الباندات في الملف أقل من المختار.")

    x = arr[[r-1, g-1, b-1], :, :].astype(np.float32)  # (3,H,W)
    H, W = x.shape[1], x.shape[2]

    valid = np.all(np.isfinite(x), axis=0)
    if nodata is not None:
        valid &= np.all(x != nodata, axis=0)

    X = x.reshape(3, -1).T  # (N,3)
    y = np.zeros((H * W,), dtype=np.uint8)  # 0 = NoData/غير مصنف

    m = valid.reshape(-1)
    if np.any(m):
        y[m] = model.predict(X[m]).astype(np.uint8)

    return y.reshape(H, W)

def plot_class_map(class_map):
    cmap = ListedColormap([
        (0, 0, 0, 0),        # 0 transparent
        (0.1, 0.4, 1.0, 1),   # 1 water
        (0.1, 0.7, 0.2, 1),   # 2 agriculture
        (0.9, 0.2, 0.2, 1),   # 3 urban
    ])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(class_map, cmap=cmap, vmin=0, vmax=3)
    ax.set_axis_off()

    from matplotlib.patches import Patch
    legend = [
        Patch(facecolor=(0.1, 0.4, 1.0, 1), label="Water (1)"),
        Patch(facecolor=(0.1, 0.7, 0.2, 1), label="Agriculture (2)"),
        Patch(facecolor=(0.9, 0.2, 0.2, 1), label="Urban (3)"),
    ]
    ax.legend(handles=legend, loc="lower center", ncol=1, frameon=True)
    return fig

def save_classified_geotiff_bytes(class_map, meta):
    out_meta = meta.copy()
    out_meta.update({
        "count": 1,
        "dtype": "uint8",
        "nodata": 0,
        "compress": "lzw"
    })

    mem = MemoryFile()
    with mem.open(**out_meta) as dst:
        dst.write(class_map.astype(np.uint8), 1)
    return mem.read()

# ---------------- UI ----------------
st.set_page_config(page_title="Raster Classification", layout="wide")

bundle = load_bundle()
model = bundle["model"]

st.sidebar.title("About")
st.sidebar.write("**Model:** Decision Tree (sklearn)")
st.sidebar.write("**Task:** Pixel-wise classification using 3 bands (RGB).")
st.sidebar.write("**Classes:**")
st.sidebar.write("- 1: Water")
st.sidebar.write("- 2: Agricultural")
st.sidebar.write("- 3: Urban")

st.title("GeoTIFF Raster Classification")

uploaded = st.file_uploader("ارفع صورة GeoTIFF متعددة الباندات", type=["tif", "tiff"])

if uploaded is None:
    st.info("ارفع ملف GeoTIFF للبدء.")
    st.stop()

try:
    arr, meta, nodata = read_geotiff(uploaded)
except Exception as e:
    st.error(f"فشل قراءة الملف. تأكد أنه GeoTIFF صالح.\n\n{e}")
    st.stop()

st.write(f"عدد الباندات: **{arr.shape[0]}** | الأبعاد: **{arr.shape[2]} x {arr.shape[1]}**")

band_options = list(range(1, arr.shape[0] + 1))
c1, c2, c3 = st.columns(3)
with c1:
    r = st.selectbox("Band for RED", band_options, index=0)
with c2:
    g = st.selectbox("Band for GREEN", band_options, index=1 if len(band_options) > 1 else 0)
with c3:
    b = st.selectbox("Band for BLUE", band_options, index=2 if len(band_options) > 2 else 0)

left, right = st.columns(2)

with left:
    st.subheader("Original (RGB preview)")
    try:
        st.image(make_rgb_preview(arr, r, g, b), use_container_width=True)
    except Exception as e:
        st.warning(f"تعذر عرض المعاينة: {e}")

run = st.button("ابدأ التصنيف")

if run:
    try:
        class_map = classify_pixels(arr, nodata, model, r, g, b)
    except Exception as e:
        st.error(str(e))
        st.stop()

    with right:
        st.subheader("Classified result")
        st.pyplot(plot_class_map(class_map), clear_figure=True)

    out_bytes = save_classified_geotiff_bytes(class_map, meta)

    st.download_button(
        "تنزيل النتيجة GeoTIFF",
        data=out_bytes,
        file_name="classified.tif",
        mime="image/tiff"
    )
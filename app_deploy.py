# -*- coding: utf-8 -*-
"""
DEPLOY MODEL AI — Deteksi Anomali LSTM-Autoencoder + PTQ (TON_IoT Network)
=========================================================================
App ini MENG-IMPOR MODEL terlatih dan MENJALANKAN INFERENSI pada data test,
lalu memvisualisasikan hasilnya. Mode utama: PREDIKSI SATU SAMPEL KONEKSI
(pilih / kirim sampel acak -> tampilkan verdict NORMAL / ANOMALI).

Bisa jalan di DUA lingkungan:
  (1) LOKAL/WSL  — baca artefak penuh dari  IDS_BASE_DIR/output_network/...
  (2) ONLINE     — baca BUNDEL kecil yang di-commit ke repo:
                     ./models/*.keras|*.tflite
                     ./sample_test.npz            (X, y, y_type contoh)
                     ./threshold_network.json     (kunci: threshold_mse)
                     ./dashboard_results.json     (angka benchmark WSL)
     Bundel disiapkan sekali di WSL dengan:  python prepare_deploy_bundle.py

Model yang di-deploy (bisa dipilih): float32 (.keras), dynamic int8 (.tflite),
static int8 (.tflite).

Menjalankan:
    pip install -r requirements_deploy.txt
    streamlit run app_deploy.py
"""
import os
import json
import pickle
import time

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# PATH — dukung mode lokal (IDS_BASE_DIR) & online (bundel repo)
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.environ.get("IDS_BASE_DIR", SCRIPT_DIR)
OUT_DIR = os.path.join(BASE_DIR, "output_network")
INTER_DIR = os.path.join(OUT_DIR, "intermediates")

# Lokasi bundel online (relatif terhadap app)
LOCAL_MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
SAMPLE_NPZ = os.path.join(SCRIPT_DIR, "sample_test.npz")
RESULTS_JSON = os.path.join(SCRIPT_DIR, "dashboard_results.json")

C_BLUE, C_GREEN, C_ORANGE, C_RED = "#2783DE", "#46A171", "#D5803B", "#E56458"
PALETTE = [C_BLUE, C_GREEN, C_ORANGE]

st.set_page_config(page_title="Deploy Model AI — Deteksi Anomali IoT",
                   page_icon="\U0001F680", layout="wide")


# ============================================================
# UTIL
# ============================================================
def read_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def fmt(v, nd=4):
    return "\u2014" if v is None else f"{v:.{nd}f}"


def file_size_kb(path):
    try:
        return os.path.getsize(path) / 1024
    except Exception:
        return None


def resolve_model_file(basename):
    """Cari model di output_network/ (lokal) atau ./models/ (bundel online)."""
    cand = [os.path.join(OUT_DIR, basename), os.path.join(LOCAL_MODELS_DIR, basename)]
    for p in cand:
        if os.path.exists(p):
            return p
    return cand[0]  # default (untuk pesan error)


MODEL_REGISTRY = {
    "float32 (.keras)": {"kind": "keras", "file": resolve_model_file("model_float32_network.keras")},
    "dynamic int8 (.tflite)": {"kind": "tflite", "file": resolve_model_file("model_dynamic_int8_network.tflite")},
    "static int8 (.tflite)": {"kind": "tflite", "file": resolve_model_file("model_static_int8_network.tflite")},
}


# ============================================================
# MUAT ARTEFAK (cached)
# ============================================================
@st.cache_data(show_spinner=False)
def load_threshold():
    # 1) pkl penuh  2) json intermediates  3) json bundel
    for path in [os.path.join(INTER_DIR, "threshold_network.pkl")]:
        try:
            with open(path, "rb") as f:
                return float(pickle.load(f)["threshold_mse"])
        except Exception:
            pass
    for path in [os.path.join(INTER_DIR, "threshold_network.json"),
                 os.path.join(SCRIPT_DIR, "threshold_network.json")]:
        j = read_json(path)
        if j and j.get("threshold_mse") is not None:
            return float(j["threshold_mse"])
    return None


@st.cache_data(show_spinner=False)
def load_test_data():
    """Kembalikan (X, y, y_type, source). Utamakan artefak penuh, fallback bundel."""
    xp = os.path.join(INTER_DIR, "X_test_network.npy")
    if os.path.exists(xp):
        X = np.load(xp).astype(np.float32)
        y = np.load(os.path.join(INTER_DIR, "y_test_network.npy"))
        try:
            yt = np.load(os.path.join(INTER_DIR, "y_type_test_network.npy"), allow_pickle=True)
        except Exception:
            yt = None
        return X, y, yt, "penuh (output_network/intermediates)"
    if os.path.exists(SAMPLE_NPZ):
        d = np.load(SAMPLE_NPZ, allow_pickle=True)
        X = d["X"].astype(np.float32)
        y = d["y"]
        yt = d["y_type"] if "y_type" in d.files else None
        return X, y, yt, "bundel contoh (sample_test.npz)"
    return None, None, None, None


@st.cache_resource(show_spinner=True)
def load_keras_model(path):
    import tensorflow as tf
    return tf.keras.models.load_model(path)


@st.cache_resource(show_spinner=True)
def load_tflite_interpreter(path):
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    return interp


# ============================================================
# INFERENSI
# ============================================================
def _errors_keras(model, X):
    Xr = model.predict(X, batch_size=256, verbose=0)
    return np.mean(np.square(X - Xr), axis=(1, 2))


def _errors_tflite(interp, X):
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    errs = np.empty(len(X), dtype=np.float64)
    for i in range(len(X)):
        interp.set_tensor(inp["index"], X[i:i + 1].astype(np.float32))
        interp.invoke()
        xr = interp.get_tensor(out["index"])
        errs[i] = float(np.mean(np.square(X[i] - xr[0])))
    return errs


def error_single(model_label, x_window):
    """Reconstruction error untuk SATU window (shape [10,44])."""
    reg = MODEL_REGISTRY[model_label]
    x = x_window[np.newaxis, :].astype(np.float32)
    if reg["kind"] == "keras":
        model = load_keras_model(reg["file"])
        return float(_errors_keras(model, x)[0])
    interp = load_tflite_interpreter(reg["file"])
    return float(_errors_tflite(interp, x)[0])


@st.cache_data(show_spinner=False)
def run_inference_batch(model_label, n_samples, seed):
    reg = MODEL_REGISTRY[model_label]
    X, y, yt, _ = load_test_data()
    n_total = len(X)
    n = min(n_samples, n_total)
    rng = np.random.RandomState(seed)
    idx = np.arange(n_total)
    if n < n_total:
        idx = np.sort(rng.choice(n_total, size=n, replace=False))
    Xs, ys = X[idx], y[idx]
    yts = yt[idx] if yt is not None else None
    t0 = time.perf_counter()
    if reg["kind"] == "keras":
        errs = _errors_keras(load_keras_model(reg["file"]), Xs)
    else:
        errs = _errors_tflite(load_tflite_interpreter(reg["file"]), Xs)
    return {"idx": idx, "errors": errs, "y": ys, "y_type": yts,
            "n": n, "n_total": n_total, "elapsed_s": time.perf_counter() - t0}


def compute_metrics(errors, y, threshold):
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 precision_score, recall_score, f1_score)
    y_pred = (errors > threshold).astype(int)
    out = {"y_pred": y_pred}
    try:
        out["roc_auc"] = float(roc_auc_score(y, errors))
        out["ap"] = float(average_precision_score(y, errors))
    except Exception:
        out["roc_auc"] = out["ap"] = None
    out["precision"] = float(precision_score(y, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y, y_pred, zero_division=0))
    out["f1"] = float(f1_score(y, y_pred, zero_division=0))
    norm = (y == 0)
    out["fpr"] = float(np.mean(y_pred[norm])) if norm.sum() else None
    return out


# ============================================================
# SIDEBAR
# ============================================================
results = read_json(RESULTS_JSON)
bench_models = {m["name"]: m for m in results.get("models", [])} if results else {}

st.sidebar.title("\U0001F680 Deploy Model AI")
model_label = st.sidebar.selectbox("Model yang di-deploy", list(MODEL_REGISTRY.keys()), index=0)
reg = MODEL_REGISTRY[model_label]
model_exists = os.path.exists(reg["file"])
sz = file_size_kb(reg["file"])
if model_exists:
    st.sidebar.success(f"{os.path.basename(reg['file'])}\n\n{sz:.1f} KB")
else:
    st.sidebar.error(f"Model tidak ditemukan:\n{reg['file']}")
st.sidebar.markdown("---")
st.sidebar.caption(f"Sumber data & model:\n`{BASE_DIR}`")

# ============================================================
# HEADER + PRA-SYARAT
# ============================================================
st.title("\U0001F680 Deploy Model AI \u2014 Deteksi Anomali Lalu Lintas IoT")
st.markdown("Model LSTM-Autoencoder (+ varian PTQ) **diimpor** lalu dijalankan pada "
            "data test. Sampel dinilai **anomali** bila *reconstruction error* > **threshold**.")

threshold = load_threshold()
X, y, yt, data_source = load_test_data()

if threshold is None:
    st.error("Threshold tidak ditemukan (threshold_network.pkl / .json).")
    st.stop()
if X is None:
    st.error("Data test tidak ditemukan. Sediakan artefak penuh (X_test_network.npy) "
             "atau bundel `sample_test.npz` (jalankan prepare_deploy_bundle.py).")
    st.stop()
if not model_exists:
    st.warning("Berkas model belum tersedia untuk pilihan ini.")
    st.stop()
try:
    import tensorflow as tf  # noqa: F401
except Exception as e:
    st.error(f"Paket **tensorflow** wajib dipasang. `pip install tensorflow`\n\n{e}")
    st.stop()

st.caption(f"Threshold: **{threshold:.6f}**  \u00b7  Data test: **{data_source}** "
           f"({len(X):,} sampel)")

tab_single, tab_batch, tab_dist, tab_type, tab_eff = st.tabs([
    "\U0001F50E Prediksi 1 Sampel",
    "\U0001F3AF Deteksi Massal",
    "\U0001F4C9 Distribusi Error",
    "\U0001F4A5 Recall per Serangan",
    "\u2696\uFE0F Efisiensi Model",
])

# ============================================================
# TAB UTAMA: PREDIKSI 1 SAMPEL
# ============================================================
with tab_single:
    st.subheader("Uji satu sampel koneksi jaringan")
    st.caption("Pilih indeks sampel test, atau tekan tombol untuk mengambil sampel acak. "
               "Model memprediksi apakah koneksi tersebut NORMAL atau ANOMALI.")

    if "sample_idx" not in st.session_state:
        st.session_state.sample_idx = 0

    col_a, col_b, col_c = st.columns([2, 1, 1])
    with col_a:
        st.session_state.sample_idx = st.number_input(
            "Indeks sampel", min_value=0, max_value=len(X) - 1,
            value=int(st.session_state.sample_idx), step=1)
    with col_b:
        if st.button("\U0001F3B2 Sampel acak"):
            st.session_state.sample_idx = int(np.random.randint(0, len(X)))
    with col_c:
        if yt is not None and st.button("\U0001F6A8 Ambil serangan"):
            atk_idx = np.where((y == 1))[0]
            if len(atk_idx):
                st.session_state.sample_idx = int(np.random.choice(atk_idx))

    i = int(st.session_state.sample_idx)
    with st.spinner("Menjalankan model pada sampel..."):
        err_i = error_single(model_label, X[i])
    pred_i = err_i > threshold
    actual_i = int(y[i]) == 1
    type_i = str(yt[i]) if yt is not None else "?"

    c1, c2, c3 = st.columns(3)
    c1.metric("Reconstruction error", f"{err_i:.6f}")
    c2.metric("Threshold", f"{threshold:.6f}")
    c3.metric("Rasio error / threshold", f"{err_i / threshold:.2f}x")

    verdict = "\U0001F6A8 ANOMALI (serangan)" if pred_i else "\u2705 NORMAL"
    truth = f"Label sebenarnya: **{'serangan' if actual_i else 'normal'}**"
    if type_i not in ("normal", "?"):
        truth += f" \u00b7 jenis: **{type_i}**"
    if pred_i == actual_i:
        st.success(f"Prediksi model ({model_label}): {verdict} \u2014 **benar**. {truth}")
    else:
        st.error(f"Prediksi model ({model_label}): {verdict} \u2014 keliru. {truth}")

    gmax = max(threshold * 3, err_i * 1.2)
    st.markdown(
        "<div style='text-align:center; font-size:22px; font-weight:600; margin-bottom:-10px;'>Reconstruction error</div>",
        unsafe_allow_html=True,
    )
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=err_i,
        number={"font": {"size": 40}},
        gauge={"axis": {"range": [0, gmax]},
               "bar": {"color": C_RED if pred_i else C_GREEN},
               "threshold": {"line": {"color": C_ORANGE, "width": 4}, "value": threshold}}))
    fig.update_layout(height=300, margin=dict(t=30, b=10, l=30, r=30))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Lihat nilai fitur window sampel ini (10 timestep \u00d7 44 fitur)"):
        st.dataframe(pd.DataFrame(X[i]).round(4), use_container_width=True)

# ============================================================
# TAB DETEKSI MASSAL
# ============================================================
with tab_batch:
    st.subheader("Deteksi pada banyak sampel sekaligus")
    n_samples = st.slider("Jumlah sampel diproses", 200, int(min(40000, len(X))),
                          int(min(3000, len(X))), step=200)
    seed = st.number_input("Seed subset", value=42, step=1)
    if st.button("\u25B6\uFE0F Jalankan deteksi massal", type="primary"):
        with st.spinner("Menjalankan inferensi..."):
            st.session_state.batch = run_inference_batch(model_label, int(n_samples), int(seed))
    res = st.session_state.get("batch")
    if res is None:
        st.info("Klik tombol untuk menjalankan deteksi massal.")
    else:
        met = compute_metrics(res["errors"], res["y"], threshold)
        y_pred = met["y_pred"]
        c = st.columns(6)
        c[0].metric("ROC-AUC", fmt(met["roc_auc"]))
        c[1].metric("Avg Precision", fmt(met["ap"]))
        c[2].metric("Precision", fmt(met["precision"]))
        c[3].metric("Recall", fmt(met["recall"]))
        c[4].metric("F1-Score", fmt(met["f1"]))
        c[5].metric("FPR", fmt(met["fpr"]))
        st.caption(f"Dihitung LIVE dari {res['n']:,} sampel ({data_source}).")
        tp = int(np.sum((y_pred == 1) & (res["y"] == 1)))
        fp = int(np.sum((y_pred == 1) & (res["y"] == 0)))
        tn = int(np.sum((y_pred == 0) & (res["y"] == 0)))
        fn = int(np.sum((y_pred == 0) & (res["y"] == 1)))
        cm = np.array([[tn, fp], [fn, tp]])
        figcm = px.imshow(cm, text_auto=True, x=["Normal", "Anomali"], y=["Normal", "Anomali"],
                          labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),
                          color_continuous_scale="Blues")
        figcm.update_layout(height=380, template="simple_white")
        st.plotly_chart(figcm, use_container_width=True)

# ============================================================
# TAB DISTRIBUSI ERROR
# ============================================================
with tab_dist:
    res = st.session_state.get("batch")
    if res is None:
        st.info("Jalankan 'Deteksi Massal' dulu untuk melihat distribusi error.")
    else:
        dfe = pd.DataFrame({"error": res["errors"],
                            "label": np.where(res["y"] == 1, "serangan", "normal")})
        fig = px.histogram(dfe, x="error", color="label", nbins=120, barmode="overlay",
                           color_discrete_map={"normal": C_BLUE, "serangan": C_RED},
                           opacity=0.65, log_y=True)
        fig.add_vline(x=threshold, line_dash="dash", line_color=C_ORANGE,
                      annotation_text=f"threshold={threshold:.4f}")
        fig.update_layout(template="simple_white", height=460,
                          xaxis_title="Reconstruction error (MSE)", yaxis_title="Jumlah (log)")
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB RECALL PER SERANGAN
# ============================================================
with tab_type:
    res = st.session_state.get("batch")
    if res is None or res["y_type"] is None:
        st.info("Jalankan 'Deteksi Massal' (dengan data ber-label jenis) untuk melihat recall per serangan.")
    else:
        y_pred = (res["errors"] > threshold).astype(int)
        yt2 = res["y_type"]
        rows = []
        for t in sorted(set(yt2.tolist())):
            if t == "normal":
                continue
            mask = (yt2 == t)
            if mask.sum():
                rows.append({"Tipe serangan": t, "Recall (live)": float(np.mean(y_pred[mask])),
                             "Jumlah": int(mask.sum())})
        if rows:
            dfr = pd.DataFrame(rows)
            fig = px.bar(dfr, x="Tipe serangan", y="Recall (live)", text="Recall (live)",
                         color_discrete_sequence=[C_GREEN])
            fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig.update_layout(template="simple_white", height=460, yaxis_range=[0, 1.05],
                              xaxis_title="", yaxis_title="Recall")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(dfr, use_container_width=True, hide_index=True)
        else:
            st.info("Tidak ada sampel serangan pada subset ini.")

# ============================================================
# TAB EFISIENSI
# ============================================================
with tab_eff:
    st.subheader("Efisiensi model")
    st.caption("Ukuran = berkas nyata di disk. Latensi resmi dari benchmark WSL (dashboard_results.json).")
    rows = []
    for label, r in MODEL_REGISTRY.items():
        bm = bench_models.get(label.split(" (")[0], {})
        rows.append({"Model": label, "Ukuran (KB)": round(file_size_kb(r["file"]) or 0, 1),
                     "Latensi WSL (ms)": bm.get("latency_ms"), "Speedup": bm.get("speedup"),
                     "Reduksi ukuran": bm.get("size_reduction")})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

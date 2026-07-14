# Dashboard Deteksi Anomali — LSTM-Autoencoder + PTQ (TON_IoT Network)

Ada **dua aplikasi** di folder ini:

| Berkas | Peran | Muat model? | Kebutuhan |
|---|---|---|---|
| `app.py` | **Viewer** — menampilkan hasil yang sudah jadi | Tidak | `requirements.txt` |
| `app_deploy.py` | **Deploy Model AI** — impor model & inferensi langsung pada data test | Ya | `requirements_deploy.txt` (butuh TensorFlow) |

Pilih `app_deploy.py` jika dosen memintamu "deploy model" (model di-import lalu
dipakai memprediksi data test). Pilih `app.py` untuk presentasi ringan/offline.

---

## A. `app_deploy.py` — Deploy Model AI (yang diminta dosen)

App ini **benar-benar memuat model terlatih** (`model_float32_network.keras`,
`model_dynamic_int8_network.tflite`, `model_static_int8_network.tflite`) lalu
**menjalankannya pada data test** (`X_test`/`y_test`/`y_type_test`) untuk
mendeteksi anomali. Sampel dengan *reconstruction error* > **threshold** dinilai
sebagai serangan.

```bash
pip install -r requirements_deploy.txt      # termasuk tensorflow
export IDS_BASE_DIR=/mnt/c/Users/nairs/Downloads/src/src_network
streamlit run app_deploy.py
```

Alur: pilih model di sidebar → atur jumlah sampel test → klik **Jalankan
inferensi**. Isi tab:
- **Hasil Deteksi** — metrik dihitung LIVE (ROC-AUC, P/R/F1, FPR) + confusion matrix + perbandingan dengan angka final WSL.
- **Distribusi Error** — histogram error normal vs serangan dengan garis threshold.
- **Recall per Serangan** — recall tiap jenis serangan dihitung live.
- **Prediksi per Sampel** — uji satu koneksi test, tampilkan verdict + gauge error.
- **Efisiensi Model** — ukuran berkas nyata + latensi resmi dari benchmark WSL.

Catatan:
- **TensorFlow wajib** (untuk `.keras` dan `.tflite` LSTM/Flex ops). Samakan versi TF dengan saat training.
- Inferensi `.tflite` berjalan per-sampel (agak berat), jadi app memakai subset test yang bisa diatur + caching. Angka final skripsi tetap dari seluruh test set.
- Latensi resmi diambil dari `dashboard_results.json` (benchmark terkendali WSL), bukan dari waktu di dalam Streamlit.
- **Mode utama**: tab **Prediksi 1 Sampel** — pilih indeks / sampel acak / ambil sampel serangan, model menampilkan verdict NORMAL vs ANOMALI + gauge error.

### Deploy online (Streamlit Community Cloud / Hugging Face Spaces)

Host tidak punya folder `output_network/` besar milikmu, jadi app bisa jalan dari
**bundel kecil**. Siapkan bundel sekali di WSL:

```bash
export IDS_BASE_DIR=/mnt/c/Users/nairs/Downloads/src/src_network
python prepare_deploy_bundle.py
```

Hasilnya folder `deploy_bundle/` berisi: `app_deploy.py`, `requirements.txt`,
`models/` (3 model, < 4 MB), `threshold_network.json`, `sample_test.npz` (contoh
data test terstratifikasi), `dashboard_results.json`, `.streamlit/config.toml`,
`packages.txt`, `.gitignore`. App otomatis membaca `models/` + `sample_test.npz`
bila artefak penuh tidak ada.

**Opsi 1 — Streamlit Community Cloud (paling mudah):**
1. Push isi `deploy_bundle/` ke sebuah repo GitHub (publik).
2. Buka share.streamlit.io → New app → pilih repo → main file `app_deploy.py`.
3. Deploy. Streamlit membaca `requirements.txt` (sudah berisi tensorflow) otomatis.

**Opsi 2 — Hugging Face Spaces:**
1. Buat Space baru → SDK **Streamlit**.
2. Upload isi `deploy_bundle/` + tambahkan header di README Space:
   ```
   ---
   title: Deteksi Anomali IoT
   sdk: streamlit
   app_file: app_deploy.py
   ---
   ```
3. Space otomatis build dari `requirements.txt`.

Catatan online: cold-start agak lama karena TensorFlow besar (~500 MB). Untuk
demo online yang ringan, model **dynamic int8** direkomendasikan.

---

## B. `app.py` — Viewer

**Viewer murni.** Dashboard ini HANYA menampilkan hasil yang sudah kamu hasilkan
di pipeline (WSL). Tidak memuat model, tidak menjalankan inferensi, tidak
mengukur latensi sendiri — sehingga loadingnya instan dan angka latensi = persis
angka yang kamu ukur di WSL.

## 1. Letak file
Taruh `app.py`, `requirements.txt`, dan `dashboard_results.json` di folder yang
memuat `output_network/`, atau atur `IDS_BASE_DIR` ke folder tersebut:
```bash
export IDS_BASE_DIR=/mnt/c/Users/nairs/Downloads/src/src_network
```

File yang dibaca:
```
output_network/intermediates/eval_float32_network.json   (dari 05_evaluate.py)
output_network/intermediates/threshold_network.json      (dari 04_threshold_v2.py)
output_network/plots/*.png                                (plot pipeline)
dashboard_results.json                                    (angka 3 model — kamu isi)
```

## 2. Isi angka perbandingan model
Buka `dashboard_results.json`, ganti `size_kb`, `latency_ms`, `roc_auc`, `f1`, dll
dengan hasil pengukuranmu di WSL. Nilai `null` = belum diukur (biarkan atau isi).

## 3. Instal & jalankan
```bash
pip install -r requirements.txt
streamlit run app.py
```
Buka http://localhost:8501

## Isi dashboard
- **Efektivitas Deteksi** — metrik float32 (dari eval json) + bar chart ROC-AUC/F1/AP antar model.
- **Efisiensi Model** — tabel + bar chart ukuran & latensi (angka WSL-mu).
- **Recall per Serangan** — dari `recall_per_type` di eval json.
- **Plot Pipeline** — menampilkan PNG yang sudah dihasilkan (ROC, PR, distribusi error, threshold).

## Catatan
- Pipeline hanya menyimpan evaluasi untuk model float32 (05_evaluate.py). Metrik &
  latensi PTQ (dynamic/static) diisi manual lewat `dashboard_results.json`.
- Kalau mau otomatis: minta dibuatkan `07_evaluate_ptq.py` yang mengevaluasi
  ketiga model sekali jalan (batch inference + latensi + metrik + recall per serangan)
  lalu menulis `dashboard_results.json` + plot PTQ secara otomatis.

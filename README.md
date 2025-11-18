# AFM Pulsed-Force Simulation**

## 🔬 AFM Pulsed-Force Simulation (Streamlit App)

This repository provides an interactive **Streamlit application** for simulating **pulsed-force atomic force microscopy (AFM)** signals.
It includes waveform generation, realistic noise injection, Savitzky–Golay smoothing, geometric contact-point detection, and interactive Plotly visualizations.

The tool is ideal for:

* understanding AFM pulsed-force dynamics
* testing CP detection methods
* visualizing the influence of noise & smoothing
* teaching AFM force-curve concepts
* prototyping CP algorithms for real AFM data

---

## ✨ **Features**

### 🔹 **1. Pulsed-force waveform simulation**

* Generates a sinusoidal tip–sample excitation signal:

  ```python
  z(t) = 0.5·cos(t + φ)
  ```
* Deflection waveform simulated using a clipped cosine model
* Supports phase offsets (`phi`) and amplitude scaling (`delta`)

### 🔹 **2. Physically motivated deflection signals**

Two models are implemented:

* `DEFLECTION`: cosine-based deflection with floor clipping
* `DEF`: intermittent-contact model with masking
* Hard clipping for indentation saturation
* Adjustable amplitude scaling

### 🔹 **3. Noise injection (SNR in dB)**

Realistic deflection noise is added using a simplified white Gaussian noise model:

```python
d_noisy = add_noise_snr(d, snr_db)
```

This lets the user explore how noise affects:

* force curves
* contact-point extraction
* slope-based metrics

### 🔹 **4. Contact point detection via geometry**

The app includes a **distance transform method**:

* Construct line from
  (P_0 = (0,0)) → (P_1 = (\text{argmax}(d), \max(d)))
* Compute perpendicular distance from each point to this line
* Maximum distance → contact point estimate

This method is simple, robust, and fast.

### 🔹 **5. Savitzky–Golay filtering**

Optional smoothing using:

```python
smooth(y, window_size, polyorder)
```

Users can interactively adjust the:

* window size
* polynomial order
* noise level

and observe how smoothing improves CP detection.

### 🔹 **6. Interactive Plotly visualizations**

Four dynamic figures:

1. **Waveforms vs Time**
2. **Force Curve (d vs z)**
3. **Noisy contact-point detection**
4. **Filtered contact-point detection**

Features include:

* magenta/purple theme
* scatter + line overlays
* dynamic vertical markers for CP
* auto-scaling based on indentation regime

### 🔹 **7. Clean Streamlit interface**

* Sliders for Delta, Phi, SNR, Gain, and Savgol parameters
* Sidebar layout with section dividers
* Dark/light mode compatible
* Responsive column layout

---

## 🚀 **How to run**

```bash
pip install -r requirements.txt
streamlit run app.py
```

Dependencies:

* Streamlit
* NumPy
* Plotly
* SciPy

---

## 🧠 **Mathematical Components**

### **Perpendicular-distance contact point**

$$
\text{dist}_i =
\frac{(x_1 - x_0)(y_i - y_0) - (y_1 - y_0)(x_i - x_0)}
{\sqrt{(x_1 - x_0)^2 + (y_1 - y_0)^2}}
$$

### **Savitzky–Golay smoothing**

Local polynomial fit:

[
y(t) = a_0 + a_1 t + a_2 t^2 + \dots
]

Smoothed point at center:
$$
\hat{y}_i = a_0
$$

---

## 📦 **Repository Structure**

```
stream_sim.py              # Streamlit application
README.md                  # You are here
.streamlit                 # config

```

---

## 🛠️ **Planned improvements**


* Hertz/Ting CP fitting methods
* Thermal noise simulator
* Multi-curve batch processing
* Export CP map to NumPy / CSV


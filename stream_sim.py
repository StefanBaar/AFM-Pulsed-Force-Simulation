import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.signal import savgol_filter

# -----------------------------
# Original functions, cleaned
# -----------------------------
def SIN(x,phi=0):
    return np.cos(x+phi)*0.5

def DEF(x,phi=0,d=0.5):


    if d<=1:
        S = d*(0.5*np.cos(x/d+phi)+0.5)
        l = 10000*(1/2-1*d/2)
        p = phi*5000/2*d/2
        if d < 0.99:
            print(l-p,l+p)
            ll = l-p
            if ll < 0:
                ll = 0

            S[ : int(round(ll,0))  ] = 0
            S[  -int(round(l+p,0)) :] = 0
    else:
        S = (0.5*np.cos(x+phi)+0.5)/d + d

    return S

def DEFLECTION(x,phi,d):
    S = (0.5*np.cos(x+phi)-0.5)+d
    S[S<0] = 0
    return S

# -----------------------------
# Noise model (SNR in dB)
# -----------------------------
def add_noise_snr(signal, snr_db):
    """Add white gaussian noise based on desired SNR in dB."""
    snr = snr_db/100
    P_signal = 0.5
    P_noise  = P_signal+0.001 / (10**(snr/10))

    noise = np.sqrt(P_noise) * np.random.normal(size=len(signal))
    return signal + snr*noise


def get_dist(za,da):
    """Get distance transformation to determine CP"""

    x01   = [-0.5, za[np.argmax(da)]]
    y01   = [   0, da.max()]
    dx    = np.diff(x01)
    dy    = np.diff(y01)
    denom = np.sqrt(dx*dx + dy*dy)

    vx = za - x01[0]
    vy = da - y01[0]

    dist = np.abs((dx * vy - dy * vx) / denom)
    return dist,[y01,x01]

def smooth(y,sw=11,sp=3):
    """Apply optional Savitzky–Golay smoothing."""
    return savgol_filter(y, sw, sp)


magenta  = "rgba(255, 0, 255, 1)"
magentat = "rgba(255, 0, 255, 0.5)"

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Pulse Force Simulation", layout="wide")

st.title("AFM Pulsed-Force Simulation")

st.sidebar.header("Controls")

# Slider: delta
delta = st.sidebar.slider(
    "Delta (0 – 3)",
    min_value = 0.0,
    max_value = 3.0,
    value     = 0.3,
    step      = 0.01,
)

# Slider: phi
phi = st.sidebar.slider(
    "Phi (-π/4 – π/4)",
    min_value = float(-np.pi/4),
    max_value = float(np.pi/4),
    value     = 0.05,
    step      = 0.001,
)



st.sidebar.write("---")
st.sidebar.write("Deflection:")
noise_db = st.sidebar.slider(
    "Noise (SNR in dB)",
    min_value = 0.0,
    max_value = 40.0,
    value     = 1.,
    step      = 0.01,
)

def_amp = st.sidebar.slider(
    "Gain",
    min_value = 0.0,
    max_value = 5.0,
    value     = 1.,
    step      = 0.01,
)

st.sidebar.write("---")
st.sidebar.write("Savgol Filter:")
ss = st.sidebar.slider(
    "window size",
    min_value = 3,
    max_value = 100,
    value     = 51,
    step      = 1,
)


s  = 1000

# -----------------------------
# Generate waveforms
# -----------------------------
t  = np.linspace(-np.pi, np.pi, s)
z  = SIN(t)


d1 = DEFLECTION(t, phi=0, d=delta)*def_amp
d2 = DEFLECTION(t, phi=phi, d=delta)*def_amp

if delta<1:
    d1[d1>1] = 1
    d2[d2>1] = 1

# -----------------------------
# Add deflection noise
# -----------------------------
d1n = add_noise_snr(d1, noise_db)
d2n = add_noise_snr(d2, noise_db)

za  =  z[:s//2]
da1 = d1[:s//2]
da2 = d2[:s//2]

da2d,dayx = get_dist(za,da2)

da1n = d1n[:s//2]
da2n = d2n[:s//2]

da2nd,danyx = get_dist(za,da2n)

sp   = 2
da1s = smooth(da1n,ss,sp)
da2s = smooth(da2n,ss,sp)

da2sd,dasyx = get_dist(za,da2s)

cp  = [da2d.max(),za[np.argmax(da2d)]]
cpn = [da2nd.max(),za[np.argmax(da2nd)]]
cps = [da2sd.max(),za[np.argmax(da2sd)]]




# -----------------------------
# Plot 1 — waveforms vs time
# -----------------------------
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=t, y=z, mode="lines", name="z(t)",line=dict(color="white")))
fig1.add_trace(go.Scatter(x=t, y=d1n, mode="lines", name="d(t), φ",line=dict(color="cornflowerblue")))
fig1.add_trace(go.Scatter(x=t[s//2:], y=d2n[s//2:], mode="lines", name="d(t), φ="+str(round(phi,2)),line=dict(color="purple")))
fig1.add_trace(go.Scatter(x=t[:s//2], y=d2n[:s//2], mode="lines", name="d(t) (appr.)",line=dict(color="magenta")))

fig1.update_layout(
    title="Waveforms vs Time",
    xaxis_title="t [2π/10k・ns] ",
    yaxis_title="U [V]",
    height=400,
    template="plotly_white"
)

# -----------------------------
# Plot 2 — Force Curve (z vs d)
# -----------------------------
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x   = z, y=d1n,
                          mode= "lines+markers",
                          marker=dict(color = "cornflowerblue",
                                      size  = 3),
                          name="d vs z, φ"))

fig2.add_trace(go.Scatter(x=z[s//2:], y=d2n[s//2:], mode="lines", name="d vs z, φ="+str(round(phi,2)),line=dict(color="purple")))
fig2.add_trace(go.Scatter(x=z[:s//2], y=d2n[:s//2], mode="lines", name="d vs z (approach.)",line=dict(color="magenta")))

fig2.update_layout(
    title="Force Curve (d vs z)",
    xaxis_title="z [V]",
    yaxis_title="d [V]",
    height=400,
    template="plotly_white",
)
if delta<=1:
    fig2.update_yaxes(range=[-0.1, 1.1])
else:
    fig2.update_yaxes(range=[-0.1, d2.max()*1.1])

fig2.update_xaxes(range=[-0.52, 0.52])


# -----------------------------
# Plot 3 — Force Curve (z vs d)
# -----------------------------
fig3 = go.Figure()
# fig3.add_trace(go.Scatter(x   = z, y=d1,
#                           mode= "lines+markers",
#                           marker=dict(color = "cornflowerblue",
#                                       size  = 3),
#                           name="d vs z, φ"))


fig3.add_trace(go.Scatter(x=za, y=da2nd, mode="markers", name="slope d. (noisy)",marker=dict(size=3,color="rgba(255,255,255,0.1)")))
fig3.add_trace(go.Scatter(x=za, y=da2d, mode="lines", name="slope distance",line=dict(color="rgba(255,255,255,0.25)")))
fig3.add_trace(go.Scatter(x=za, y=da2n , mode="markers", name="d vs z (noisy)",marker=dict(size=3,color=magentat)))
fig3.add_trace(go.Scatter(x=za, y=da2 , mode="lines", name="d vs z (approach.)",line=dict(color=magenta)))

fig3.add_trace(go.Scatter(x=[cp[1],cp[1]], y=[0,cp[0]] , mode="lines+markers", name="cp="+str(round(cp[1],2))+"V",line=dict(dash="dash",width=1,color="red")))
fig3.add_trace(go.Scatter(x=[cpn[1],cpn[1]], y=[0,cpn[0]] , mode="lines+markers", name="cp_n="+str(round(cpn[1],2))+"V",line=dict(dash="dash",width=1,color="white")))

fig3.add_trace(go.Scatter(x=danyx[1], y=danyx[0] , mode="lines+markers",name="ref. sl.",line=dict(dash="dash",width=1,color="cornflowerblue")))

#fig3.add_vline(x=za[np.argmax(da2d)], line_width=2, line_dash="dash", line_color="white")

fig3.update_layout(
    title="Contact Point (d vs z)",
    xaxis_title="z [V]",
    yaxis_title="d [V]",
    height=400,
    template="plotly_white",
)
if delta<=1:
    fig3.update_yaxes(range=[-0.1, 1.1])
else:
    fig3.update_yaxes(range=[-0.1, d2.max()*1.1])

fig3.update_xaxes(range=[-0.52, 0.52])





fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=za, y=da2sd, mode="markers", name="slope d. (n.filter)",marker=dict(size=3,color="rgba(255,255,255,0.1)")))
fig4.add_trace(go.Scatter(x=za, y=da2d , mode="lines", name="slope distance",line=dict(color="rgba(255,255,255,0.25)")))
fig4.add_trace(go.Scatter(x=za, y=da2s , mode="markers", name="d vs z (n.filter)",marker=dict(size=3,color=magentat)))
fig4.add_trace(go.Scatter(x=za, y=da2  , mode="lines", name="d vs z (appr.)",line=dict(color=magenta)))


fig4.add_trace(go.Scatter(x=[cp[1],cp[1]], y=[0,cp[0]] , mode="lines+markers", name="cp="+str(round(cp[1],2))+"V",line=dict(dash="dash",width=1,color="red")))
fig4.add_trace(go.Scatter(x=[cps[1],cps[1]], y=[0,cps[0]] , mode="lines+markers", name=r"cp_f="+str(round(cps[1],2))+"V",line=dict(dash="dash",width=1,color="white")))

fig4.add_trace(go.Scatter(x=dasyx[1], y=dasyx[0] , mode="lines+markers",name="ref. sl.",line=dict(dash="dash",width=1,color="cornflowerblue")))

fig4.update_layout(
    title="CP Filtered (d vs z)",
    xaxis_title="z [V]",
    yaxis_title="d [V]",
    height=400,
    template="plotly_white",
)
if delta<=1:
    fig4.update_yaxes(range=[-0.1, 1.1])
else:
    fig4.update_yaxes(range=[-0.1, d2.max()*1.1])

fig4.update_xaxes(range=[-0.52, 0.52])





col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig1, use_container_width=True)
with col2:
    st.plotly_chart(fig2, use_container_width=True)

col3, col4 = st.columns(2)
with col3:
    st.plotly_chart(fig3, use_container_width=True)
with col4:
    st.plotly_chart(fig4, use_container_width=True)


col5, col6 = st.columns(2)
with col5:
    st.markdown(r"""
    ---
    # 1. Geometry")
    * $x =$ z (piezo position)
    * $y =$ d (deflection)

    "Endpoints:")

     * $P_0 = (x_0, y_0) = (0,0)$
     * $P_1 = (x_1, y_1) = (argmax(y), max(y))$



    For any point ($P_i = (x_i, y_i)$), the **signed perpendicular distance** from the line ($P_0 P_1$) is:

    $$\text{dist}_i = \frac{(x_1 - x_0)(y_i - y_0) - (y_1 - y_0)(x_i - x_0)}{\sqrt{(x_1 - x_0)^2 + (y_1 - y_0)^2}}$$

     * The numerator is the 2D cross product (gives sign).
     * The denominator is just the line length.
    ---

    """)


with col6:
    st.markdown(r"""

    ---

    # Savitzky–Golay Filter Works

    Consider your data as:

    y₁, y₂, ..., yₙ


    You choose:

    - **Window length** \( W \) (odd number, e.g. 11, 51, 101 …)
    - **Polynomial order** \( p \) (e.g. 2 or 3)

    For every point \( i \), take a window of \( W \) neighboring points:

    $$
    y_{i-k}, \ldots, y_{i+k} \quad (W = 2k + 1)
    $$

    Inside that local window, fit a polynomial:

    $$
    y(t) = a_0 + a_1 t + a_2 t^2 + \dots + a_p t^p
    $$

    using **least squares**.

    Then evaluate the polynomial at the **center** of the window:

    $$
    \hat{y}_i = a_0
    $$

    This gives the smoothed value for point \( i \).

    ---

    """)

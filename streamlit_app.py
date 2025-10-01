import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO

# =========================
# UI & PAGE CONFIG
# =========================
st.set_page_config(page_title="EDIE v9/v10 — Extended Drake–Information Explorer", layout="wide")
st.title("Extended Drake–Information Equation — v9 / v10 Explorer")
st.caption("MD14 SFRD with MLP-like uplift, cosmology-aware volume, v9 kernels (bio/tech/detectability), and v10 Λ tuning.")

# =========================
# SIDEBAR CONTROLS
# =========================
with st.sidebar:
    st.header("Cosmology (v10)")
    H0 = st.number_input("H0 (km/s/Mpc)", 40.0, 90.0, 70.0, 0.5)
    Om = st.slider("Ωm", 0.05, 0.6, 0.30, 0.01)
    Ol = st.slider("ΩΛ", 0.0, 1.2, 0.70, 0.01)   # curvature: Ωk = 1 - Ωm - ΩΛ
    z_max = st.slider("z max", 0.2, 10.0, 3.0, 0.1)
    n_pts = st.slider("# z samples", 50, 2500, 700, 50)

    st.header("SFR Kernel (MD14 + uplift)")
    A  = st.number_input("A (norm) [arb Msun/yr/Mpc³]", 0.0001, 0.2, 0.015, 0.0001)
    a_ = st.number_input("α (rise power)", 0.5, 6.0, 2.7, 0.1)
    b_ = st.number_input("β (scale for turnover)", 1.0, 6.0, 2.9, 0.1)
    g_ = st.number_input("γ (fall power)", 1.0, 12.0, 5.6, 0.1)
    use_uplift = st.checkbox("Apply MLP-like uplift", True)
    umax = st.slider("Max uplift factor", 1.0, 3.0, 1.5, 0.05)
    z1   = st.slider("uplift start z1", 0.0, 5.0, 0.7, 0.05)
    z2   = st.slider("uplift end z2",   0.2, 6.0, 2.2, 0.05)
    sharp= st.slider("uplift sharpness", 0.5, 12.0, 3.0, 0.1)

    st.header("v9: Mass & Metallicity")
    M_mu = st.slider("log10 M* mean (M☉)", 8.0, 12.5, 10.5, 0.1)
    M_sig= st.slider("log10 M* σ", 0.05, 1.5, 0.5, 0.05)
    Z_floor = st.slider("Metallicity floor (Z/Z☉)", 0.01, 1.0, 0.20, 0.01)

    st.header("v9: Bio Emergence (Deamer ramp)")
    t0_Gyr = st.number_input("Universe age t0 (Gyr)", 10.0, 20.0, 13.8, 0.1)
    tau_abiog = st.slider("Abiogenesis timescale τ (Gyr)", 0.1, 10.0, 1.0, 0.1)
    t_lag = st.slider("Lag to technology (Gyr)", 0.0, 8.0, 1.0, 0.1)

    st.header("v9: Culture Survival (birth–death + ceiling)")
    birth = st.slider("Birth rate b", 0.0, 3.0, 0.6, 0.05)
    death = st.slider("Death rate d", 0.0, 3.0, 0.4, 0.05)
    cap   = st.slider("Resource ceiling K (arb)", 0.1, 20.0, 2.0, 0.1)

    st.header("v9: Detectability")
    eps_up = st.slider("ε_up (duty/usage)", 0.0, 1.0, 0.5, 0.05)
    eps_waste = st.slider("ε_waste (IR waste-heat)", 0.0, 0.5, 0.05, 0.01)
    p_dimm = st.slider("Dimming exponent p", 0.0, 6.0, 2.0, 0.1)
    S_min = st.slider("Survey floor S_min", 0.0, 5.0, 1.0, 0.05)

    st.header("Optional: IR K-scaler & Time Gate")
    use_KIR = st.checkbox("Include IR K-scaler", False)
    K0 = st.slider("K_IR amplitude K₀", 0.0, 3.0, 1.0, 0.05)
    lam_obs_um = st.slider("Instrument λ_obs (μm)", 0.5, 50.0, 4.5, 0.1)
    lam_peak_um= st.slider("Emitter λ_peak (μm at rest)", 1.0, 80.0, 10.0, 0.5)
    k_width = st.slider("K width σ (μm)", 0.1, 20.0, 4.0, 0.1)

    use_time_gate = st.checkbox("Include time-window gate W_time", False)
    t_obs_Gyr = st.number_input("Observer time t_f (Gyr)", 8.0, 20.0, 13.8, 0.1)
    tau_gate = st.slider("Tech lag threshold (Gyr, gate)", 0.0, 8.0, 1.0, 0.1)

    st.header("Sweeps")
    sweep_kind = st.selectbox("Sweep variable", ["None", "ε_waste", "z_max", "ΩΛ"])
    n_sweep = st.slider("# sweep points", 2, 20, 6, 1)

# =========================
# COSMOLOGY HELPERS (v10, curvature-aware)
# =========================
c_km_s = 299792.458

def Ez(z, Om, Ol):
    Ok = 1.0 - Om - Ol
    return np.sqrt(Om*(1+z)**3 + Ok*(1+z)**2 + Ol)

def Hz(z, H0, Om, Ol):
    return H0 * Ez(z, Om, Ol)

def cumtrapz_np(y, x, initial=0.0):
    y = np.asarray(y); x = np.asarray(x)
    areas = 0.5*(y[1:] + y[:-1])*(x[1:] - x[:-1])
    return np.concatenate(([initial], np.cumsum(areas)))

def comoving_distance(z_grid, H0, Om, Ol):
    z = np.asarray(z_grid)
    integrand = c_km_s / Hz(z, H0, Om, Ol)
    return cumtrapz_np(integrand, z, initial=0.0)

def transverse_comoving_distance(Dc, H0, Om, Ol):
    Ok = 1.0 - Om - Ol
    if np.isclose(Ok, 0.0):
        return Dc
    sqrtOk = np.sqrt(abs(Ok))
    chi = (H0 / c_km_s) * sqrtOk * Dc           # dimensionless
    if Ok > 0:  # open
        return (c_km_s / (H0*sqrtOk)) * np.sinh(chi)
    else:       # closed
        return (c_km_s / (H0*sqrtOk)) * np.sin(chi)

def dVc_dz_per_sr(z_grid, H0, Om, Ol):
    z = np.asarray(z_grid)
    Dc = comoving_distance(z, H0, Om, Ol)
    DM = transverse_comoving_distance(Dc, H0, Om, Ol)
    return (c_km_s / Hz(z, H0, Om, Ol)) * (DM**2)

# =========================
# SFR & UPLIFT
# =========================
def md14_psi(z, A=0.015, a=2.7, b=2.9, g=5.6):
    z = np.asarray(z)
    return A * (1+z)**a / (1 + ((1+z)/b)**g)

def uplift_factor(z, umax=1.5, z1=0.7, z2=2.2, k=3.0):
    s = lambda x: 1.0/(1.0+np.exp(-x))
    raw = np.clip(s(k*(z - z1)) * s(-k*(z - z2)), 0.0, 1.0)
    return 1.0 + (umax - 1.0)*raw

def psi_fused(z, A, a, b, g, apply_uplift=True, umax=1.5, z1=0.7, z2=2.2, k=3.0):
    base = md14_psi(z, A, a, b, g)
    return base * (uplift_factor(z, umax, z1, z2, k) if apply_uplift else 1.0)

# =========================
# v9 KERNELS
# =========================
def R_MZ(z, Mmu=10.5, Msig=0.5, Zfloor=0.2):
    # Metallicity proxy: monotonic decline with z, floored
    Zz = np.clip(1.0/(1.0 + 0.7*z), Zfloor, 1.0)
    # Lognormal mass moment boost ~ exp((σ ln 10)^2 / 2)
    mass_boost = np.exp((Msig*np.log(10))**2 / 2.0)
    return Zz * mass_boost

def lookback_time_Gyr(z, H0=70.0, Om=0.3, Ol=0.7):
    if z <= 0: 
        return 0.0
    zz = np.linspace(0.0, z, 800)
    integrand = 1.0 / ((1.0+zz) * Ez(zz, Om, Ol))  # 1/((1+z)E(z))
    val = np.trapz(integrand, zz)
    h = H0/100.0
    return (9.778 / h) * val  # Gyr (approx standard conversion)

def B_hard(z, t0=13.8, tau=1.0, t_lag=1.0, H0=70.0, Om=0.3, Ol=0.7):
    t_lb = lookback_time_Gyr(z, H0, Om, Ol)
    t_form = t0 - t_lb
    t_star = max(t0 - t_form - t_lag, 0.0)
    return 1.0 - np.exp(-t_star / max(tau, 1e-6))

def C_bd(z, b=0.6, d=0.4, K=2.0):
    net = b - d
    A = K/(1.0+K)  # maps ceiling to (0,1)
    # logistic in (b - d), centered near 0.2 for illustrative survival threshold
    P = 1.0 / (1.0 + np.exp(-5.0*(net - 0.2)))
    return P * A

def D_detect(z, eps_up=0.5, eps_waste=0.05, p=2.0, Smin=1.0):
    return eps_up * eps_waste / ((1.0+z)**p + Smin)

def K_IR(z, K0=1.0, lam_obs=4.5, lam_peak=10.0, width=4.0):
    lam_emit_obs = lam_peak * (1.0 + z)
    return K0 * np.exp(-0.5 * ((lam_emit_obs - lam_obs) / max(width, 1e-6))**2)

def W_time_gate(z, t_obs=13.8, tau_gate=1.0, H0=70.0, Om=0.3, Ol=0.7):
    # 1 if enough time since formation to reach tech (simple gate)
    t_lb = lookback_time_Gyr(z, H0, Om, Ol)
    t_form = t0_Gyr - t_lb
    return 1.0 if (t_obs - t_form) >= tau_gate else 0.0

# =========================
# COMPUTE v9 INTEGRAND & N(t0)
# =========================
z = np.linspace(0.0, z_max, int(n_pts))

# Geometry
dVc = dVc_dz_per_sr(z, H0, Om, Ol)   # per sr

# SFR fused
psi = psi_fused(z, A, a_, b_, g_, apply_uplift=use_uplift, umax=umax, z1=z1, z2=z2, k=sharp)
upl = uplift_factor(z, umax, z1, z2, sharp) if use_uplift else np.ones_like(z)

# v9 factors
RMZ = R_MZ(z, Mmu=M_mu, Msig=M_sig, Zfloor=Z_floor)
Bz  = np.array([B_hard(zz, t0=t0_Gyr, tau=tau_abiog, t_lag=t_lag, H0=H0, Om=Om, Ol=Ol) for zz in z])
Cz  = C_bd(z, b=birth, d=death, K=cap)
Dz  = D_detect(z, eps_up=eps_up, eps_waste=eps_waste, p=p_dimm, Smin=S_min)

# Optional IR K-scaler and time gate
if use_KIR:
    Kz = K_IR(z, K0=K0, lam_obs=lam_obs_um, lam_peak=lam_peak_um, width=k_width)
else:
    Kz = np.ones_like(z)

if use_time_gate:
    Wz = np.array([W_time_gate(zz, t_obs=t_obs_Gyr, tau_gate=tau_gate, H0=H0, Om=Om, Ol=Ol) for zz in z])
else:
    Wz = np.ones_like(z)

# v9 integrand (per sr, arbitrary normalization)
integrand_v9 = dVc * psi * RMZ * Bz * Cz * Dz * Kz * Wz
N_v9 = np.trapz(integrand_v9, z)

# =========================
# DISPLAY / PLOTS
# =========================
colA, colB, colC = st.columns(3)
colA.metric("N(t₀) per sr (arb)", f"{N_v9:,.3e}")
colB.metric("Ωk (curvature)", f"{1.0-Om-Ol:+.3f}")
colC.metric("z grid size", f"{len(z)}")

def plot_line(x, ys, labels, title, ylab):
    fig, ax = plt.subplots()
    for y, lab in zip(ys, labels):
        ax.plot(x, y, label=lab)
    ax.set_xlabel("z"); ax.set_ylabel(ylab); ax.set_title(title); ax.legend()
    st.pyplot(fig)

plot_line(z, [psi], ["ψ_fused"], "SFR Kernel (fused)", "ψ(z) [arb]")
plot_line(z, [upl], ["uplift factor"], "MLP-like uplift factor", "×")
plot_line(z, [RMZ, Bz, Cz, Dz*Kz*Wz], ["R_MZ","B_hard","C_bd","D·K·W"], "v9 Factors", "×")
plot_line(z, [integrand_v9], ["Integrand v9"], "Drake–Information Integrand (v9)", "arb per sr")

# Export integrand CSV
csv = np.column_stack([z, dVc, psi, RMZ, Bz, Cz, Dz, Kz, Wz, integrand_v9])
buf = BytesIO()
np.savetxt(buf, csv, delimiter=",",
           header="z,dVc,psi_fused,R_MZ,B_hard,C_bd,D,K_IR,W_time,integrand_v9",
           comments="")
st.download_button("Download integrand CSV", buf.getvalue(),
                   file_name="edie_v9_integrand.csv", mime="text/csv")

# =========================
# SWEEPS
# =========================
rows = []
if sweep_kind != "None":
    if sweep_kind == "ε_waste":
        vals = np.linspace(0.0, 0.2, n_sweep)
        for ew in vals:
            Dz_s = D_detect(z, eps_up=eps_up, eps_waste=ew, p=p_dimm, Smin=S_min)
            I = dVc * psi * RMZ * Bz * Cz * Dz_s * Kz * Wz
            rows.append({"eps_waste": ew, "N": np.trapz(I, z)})
    elif sweep_kind == "z_max":
        vals = np.linspace(max(0.5, z_max/4), z_max, n_sweep)
        for zcut in vals:
            m = z <= zcut
            rows.append({"z_max": zcut, "N": np.trapz(integrand_v9[m], z[m])})
    elif sweep_kind == "ΩΛ":
        vals = np.linspace(max(0.0, Ol-0.3), Ol+0.3, n_sweep)
        for Ol_s in vals:
            dVc_s = dVc_dz_per_sr(z, H0, Om, Ol_s)
            # recompute time-sensitive terms if gate enabled
            if use_time_gate:
                Wz_s = np.array([W_time_gate(zz, t_obs=t_obs_Gyr, tau_gate=tau_gate, H0=H0, Om=Om, Ol=Ol_s) for zz in z])
            else:
                Wz_s = Wz
            # B_hard depends on Ol via lookback time
            Bz_s = np.array([B_hard(zz, t0=t0_Gyr, tau=tau_abiog, t_lag=t_lag, H0=H0, Om=Om, Ol=Ol_s) for zz in z])
            I = dVc_s * psi * RMZ * Bz_s * Cz * Dz * Kz * Wz_s
            rows.append({"Omega_Lambda": Ol_s, "N": np.trapz(I, z)})

if rows:
    df = pd.DataFrame(rows)
    st.subheader("Sweep Results")
    st.dataframe(df)
    xcol = df.columns[0]
    fig, ax = plt.subplots()
    ax.plot(df[xcol].values, df["N"].values, marker="o")
    ax.set_xlabel(xcol); ax.set_ylabel("N (arb per sr)"); ax.set_title(f"Sweep: {xcol} → N")
    st.pyplot(fig)
    st.download_button("Download sweep CSV", df.to_csv(index=False).encode(),
                       file_name="edie_v9_sweep.csv", mime="text/csv")

st.markdown("---")
st.markdown(
    r"""
**EDIE v9** form:
\[
N(t_0)=\int_{z_{\min}}^{z_{\max}}\!\int \big[ \psi_{\rm fused}(\hat n,z)\; p(M,Z|z)\; \mathcal{B}_{\rm hard}(t_\star)\; \mathcal{C}_{\rm bd}(t_\star)\big]\; \mathcal{D}(\hat n,z;\epsilon_{\rm up})\; \frac{dV_c}{dz\,d\Omega}\, dz\, d\Omega\, dM
\]
with \(dV_c/(dz\,d\Omega)= \frac{c}{H(z)} D_M^2(z)\) and \(H(z)=H_0\,E(z)\) for general \(\Omega_m,\Omega_\Lambda,\Omega_k\).
All factors here are illustrative and tunable; units are in arbitrary normalization.
"""
)
st.caption("Data-friendly, NumPy-only build. No endorsements implied. NASA/ESA/CSA public data allowed with attribution.")

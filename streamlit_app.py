import streamlit as st
import numpy as np
import math
from scipy.integrate import cumtrapz, odeint
import pandas as pd
import plotly.express as px
from io import BytesIO

# App Title
st.set_page_config(page_title="Extended Drake UI", layout="wide", initial_sidebar_state="expanded")
st.title("🪐 Extended Drake–Information Equation Navigator")
st.markdown("Interactive MC sweeps for N(t_0). Adjust params, run, explore N(z).")

# Sidebar: Params & Controls
st.sidebar.header("Parameters")
z_max = st.sidebar.slider("z_max", 0.5, 3.0, 2.0, 0.1)
epsilon_waste = st.sidebar.slider("ε_waste (fused prior)", 0.01, 0.5, 0.05, 0.01)
k_max = st.sidebar.slider("K_max", 0.8, 2.0, 1.1, 0.1)
sfr_base = st.sidebar.selectbox("SFR Base", ["Lognormal + Rising Hybrid", "MD14 Double Power-Law"])
n_samples = st.sidebar.slider("MC Samples", 100, 1000, 500, 100)
run_button = st.sidebar.button("Run Sweep")

# Chat Input
chat_prompt = st.sidebar.text_input("Chat Query (e.g., 'sweep z=1.5')")

# Cosmology Fallback
H0 = 70.0 * 1000.0 / 3.085677581e22
Om0, Ol0 = 0.3, 0.7
c = 299792458.0

def Ez(z): return math.sqrt(Om0*(1+z)**3 + Ol0)
def dc_comoving(z, steps=1024):
    zgrid = np.linspace(0.0, z, steps)
    integrand = 1.0 / np.array([Ez(zz) for zz in zgrid])
    return (c / H0) * np.trapz(integrand, zgrid)
def dL(z): return (1+z) * dc_comoving(z)

# Hard-Steps Bio
def B_hardsteps(t_star, r0=0.1, tau=0.05, alpha=2):
    if np.isscalar(t_star):
        t_star = np.array([t_star])
    t_grid = np.linspace(0, t_star.max(), 100)
    lambda_chem = r0 * t_grid

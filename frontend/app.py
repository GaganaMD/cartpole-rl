import streamlit as st
import requests
import numpy as np

API_URL = "http://127.0.0.1:8000"

st.title("CartPole RL Agent (FastAPI + PyTorch)")

st.markdown("Send a CartPole state to the RL policy and get the chosen action + probabilities.")

# Manual state sliders
cart_pos = st.slider("Cart Position", -2.4, 2.4, 0.0, 0.01)
cart_vel = st.slider("Cart Velocity", -3.0, 3.0, 0.0, 0.01)
pole_angle = st.slider("Pole Angle", -0.2, 0.2, 0.0, 0.01)
pole_vel = st.slider("Pole Angular Velocity", -3.0, 3.0, 0.0, 0.01)

state = [cart_pos, cart_vel, pole_angle, pole_vel]
st.write("Current state:", state)

if st.button("Query Agent"):
    try:
        resp = requests.post(f"{API_URL}/act", json=state, timeout=5)
        data = resp.json()
        st.success(f"Action: {data['action']}  (0 = left, 1 = right)")
        st.write("Probabilities:", np.round(data["probs"], 3))
    except Exception as e:
        st.error(f"Error talking to backend: {e}")

st.caption("Backend: FastAPI @ /act • Model: PyTorch policy trained on CartPole-v1")

import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt

#API_URL = "http://127.0.0.1:8000"  # change to "http://backend:8000" when inside Docker compose

API_URL = "https://cartpole-rl.onrender.com"



st.title("CartPole RL Agent (FastAPI + PyTorch)")
st.markdown("Interact with a trained CartPole policy served via FastAPI.")

st.subheader("Query action for a single state")

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

st.markdown("---")
st.subheader("Run one full episode")

# choose policy
policy = st.radio(
    "Policy to use",
    options=["trained", "random"],
    index=0,
    help="Compare the trained PPO agent with a random policy.",
)

steps = st.slider("Max steps", min_value=50, max_value=500, value=200, step=10)

if st.button("Run Episode with current policy"):
    try:
        resp = requests.post(
            f"{API_URL}/simulate",
            params={"steps": steps, "policy": policy},
            timeout=15,
        )
        data = resp.json()
        rewards = data.get("rewards", [])
        total_reward = data.get("total_reward", 0.0)
        steps_used = data.get("steps", 0)
        used_policy = data.get("policy", policy)

        st.write(
            f"Policy: **{used_policy}** — "
            f"Total reward: {total_reward:.1f} over {steps_used} steps"
        )

        if rewards:
            fig, ax = plt.subplots()
            ax.plot(rewards)
            ax.set_xlabel("Step")
            ax.set_ylabel("Reward")
            ax.set_title("Per-step reward")
            st.pyplot(fig)
    except Exception as e:
        st.error(f"Error running episode: {e}")

st.caption("Backend: FastAPI • Model: PyTorch policy trained on CartPole-v1 • Frontend: Streamlit")

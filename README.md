# 🎯 CartPole RL – End-to-End Deployed PPO Agent

![CI](https://github.com/GaganaMD/cartpole-rl/actions/workflows/ci.yml/badge.svg?branch=main)



Interactive CartPole demo built with **PyTorch**, **FastAPI**, **Streamlit**, and **Docker**.

Train a PPO agent, serve it behind an API, and explore its behavior in a web UI.

---

## 🚀 Features

* ✅ PPO-trained CartPole agent implemented in PyTorch
* ✅ FastAPI backend exposing:

  * `POST /act` – greedy action + probabilities for a given CartPole state
  * `POST /simulate` – run an episode with the current policy and return rewards
  * `GET /health` – health check endpoint
  * `GET /metrics` – basic runtime metrics
* ✅ Streamlit frontend to:

  * Probe the policy with manual state sliders
  * Run full episodes and visualize rewards over time
* ✅ Fully Dockerized frontend + backend with `docker-compose` for one-command startup

> This project is an end-to-end RL deployment example and a strong foundation for adding production-grade MLOps components such as logging, monitoring, CI/CD, and experiment tracking.

---

## 🏗️ Architecture

```
[ PyTorch PPO Model ]
          │
          ▼
[ FastAPI Backend ]  ← uvicorn (serves /act and /simulate)
          │
          ▼
[ Streamlit UI ]     ← calls backend, visualizes rewards
          │
          ▼
[ User Browser ]
```

All services run together using **Docker Compose**.

---

# 🛠️ Getting Started (Local Development)

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/cartpole-rl.git
cd cartpole-rl
```

---

## 2️⃣ Backend – FastAPI

### Create and Activate a Python Environment

Using `venv`:

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows
```

Or use Conda if preferred.

### Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### Run the API Server

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Open API Docs

Visit:

```
http://127.0.0.1:8000/docs
```

Available endpoints:

* `POST /act`
* `POST /simulate`
* `GET /health`
* `GET /metrics`

---

## 3️⃣ Frontend – Streamlit

Open a new terminal and activate the same (or a separate) environment.

### Install Frontend Dependencies

```bash
cd frontend
pip install -r requirements.txt
```

### Run Streamlit

```bash
streamlit run app.py --server.port 8501
```

### Open the Web UI

```
http://127.0.0.1:8501
```

You can:

* Send custom states to `/act` and inspect action probabilities.
* Run full episodes via `/simulate`.
* Visualize reward curves interactively.

---

# 🐳 Running with Docker Compose

To run both backend and frontend in containers:

```bash
cd cartpole-rl
docker compose up --build
```

This will:

* Build the backend image with the trained PPO model included.
* Build the Streamlit frontend image.
* Start both services on a shared Docker network.

### Access Services

* FastAPI Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* Streamlit UI: [http://127.0.0.1:8501](http://127.0.0.1:8501)

### Stop the Stack

```bash
docker compose down
```

---

# 📌 Project Status

## ✅ Current Capabilities

* End-to-end RL inference pipeline (Model → API → UI)
* Containerized deployment with Docker + Compose
* Basic runtime observability via `/health` and `/metrics`

---

# 🔮 Planned Enhancements

* 🔁 Random vs Trained policy toggle in UI and backend
* 🗃️ SQLite logging of episodes
* 📊 "Recent Runs" table in the frontend
* 🌍 Deployment to VPS / EC2 behind reverse proxy (Caddy or Nginx) with HTTPS
* 🔄 CI/CD using GitHub Actions for automated Docker builds and deployment

---

# 📦 MLOps Perspective

This project demonstrates several practical MLOps principles:

### ✔ Model Packaging & Serving

* PPO model wrapped inside a FastAPI service
* Inference exposed via REST endpoints

### ✔ Reproducible Environments

* Dockerized services
* Infrastructure defined declaratively with `docker-compose`

### ✔ Observability

* Health endpoint
* Metrics endpoint

---

## 🚀 Turning This Into a Full MLOps Case Study

To elevate this into a production-ready MLOps example:

* Add experiment tracking (e.g., MLflow or Weights & Biases)
* Introduce model versioning
* Implement structured logging
* Add monitoring for:

  * Episode reward distributions
  * API latency
  * Failure rates
* Set up automated testing + CI/CD pipelines

---

# 🧰 Tech Stack

* **Python**
* **PyTorch**
* **Gymnasium (CartPole-v1)**
* **FastAPI**
* **Uvicorn**
* **Streamlit**
* **Docker**
* **docker-compose**

---

# 👩‍💻 Author

Built as an end-to-end Reinforcement Learning deployment project.

You can extend this into a production-grade RL serving platform or use it as a template for deploying other deep learning models.

---

⭐ If you found this useful, consider starring the repository!

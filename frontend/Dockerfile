FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir streamlit requests numpy

# Copy the single app.py into /app
COPY app.py /app/app.py

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

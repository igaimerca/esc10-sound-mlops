import random
from pathlib import Path

from locust import HttpUser, between, task

CLIPS = sorted(Path("data/test").rglob("*.wav"))
PAYLOADS = [(p.name, p.read_bytes()) for p in CLIPS]


class SoundClassifierUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(10)
    def predict(self):
        name, data = random.choice(PAYLOADS)
        self.client.post("/api/predict", files={"file": (name, data, "audio/wav")}, name="/api/predict")

    @task(1)
    def health(self):
        self.client.get("/api/health", name="/api/health")

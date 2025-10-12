from locust import HttpUser, task, between
import json

class PenguinUser(HttpUser):
    wait_time = between(1, 3)  # Espera entre 1 y 3 segundos entre requests
    
    @task(1)
    def predict_penguin(self):
        """Prueba el endpoint de predicción"""
        payload = {
            "bill_length_mm": 39.1,
            "bill_depth_mm": 18.7,
            "flipper_length_mm": 181.0,
            "body_mass_g": 3750.0,
            "year": 2007,
            "sex_Female": 0,
            "sex_Male": 1,
            "island_Biscoe": 0,
            "island_Dream": 0,
            "island_Torgersen": 1
        }
        self.client.post("/predict", json=payload)
    
    @task(1)
    def health_check(self):
        """Prueba el endpoint de health"""
        self.client.get("/health")
    
    @task(2)
    def model_info(self):
        """Prueba el endpoint de info del modelo"""
        self.client.get("/model-info")
import requests
import json

class FogService:
    def __init__(self, edge_url="http://edge-service:8002"):
        self.edge_url = edge_url

    def process_data(self, data):
        try:
            # Preprocess data if needed
            processed_data = self.preprocess(data)
            
            # Forward to edge service
            response = requests.post(
                f"{self.edge_url}/predict",
                json={"data": processed_data}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Edge service error: {response.status_code}"}
        
        except Exception as e:
            return {"error": f"Fog service error: {str(e)}"}

    def preprocess(self, data):
        # Add any necessary preprocessing steps
        return data
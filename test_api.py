from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_predict_endpoint():
    response = client.get("/predict?user_id=435505")
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 0
    print("✅ API test OK")

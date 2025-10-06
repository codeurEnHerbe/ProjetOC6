from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_predict_endpoint():
    response = client.get("/predict?user_id=100002")
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 1
    assert abs(data["probability"] - 0.9826291149297616) < 1e-6
    print("✅ API test OK")

import unittest
import json
import pandas as pd
from api import app

class TestSystem(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_01_health_endpoint(self):
        """Test health check GET / and GET /health."""
        for path in ["/", "/health"]:
            res = self.client.get(path)
            self.assertEqual(res.status_code, 200)
            data = json.loads(res.data)
            self.assertEqual(data.get("status"), "ok")
        print("\n[PASS] Test 1: Flask GET / and GET /health endpoints are healthy.")

    def test_02_metrics_and_logs(self):
        """Test GET /metrics and GET /logs endpoints."""
        m_res = self.client.get("/metrics")
        self.assertEqual(m_res.status_code, 200)

        l_res = self.client.get("/logs")
        self.assertEqual(l_res.status_code, 200)
        print("[PASS] Test 2: Flask GET /metrics and GET /logs endpoints are functional.")

    def test_03_predict_with_real_guide_record(self):
        """Test prediction POST /predict with a real record from GUIDE_Test.csv."""
        df = pd.read_csv("dataset/GUIDE_Test.csv", nrows=1, low_memory=False)
        raw_record = df.iloc[0].to_dict()
        clean_record = {k: v for k, v in raw_record.items() if pd.notna(v)}

        payload = {"record": clean_record}
        res = self.client.post("/predict", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(res.status_code, 200)

        data = json.loads(res.data)
        self.assertTrue(data.get("success"))
        self.assertIn(data.get("prediction"), ["TruePositive", "BenignPositive", "FalsePositive"])
        self.assertIn("probabilities", data)
        print(f"[PASS] Test 3: Real GUIDE record prediction -> Predicted Grade: {data.get('prediction')}")

    def test_04_missing_input(self):
        """Test error handling when no JSON is provided."""
        res = self.client.post("/predict", data=json.dumps({}), content_type="application/json")
        self.assertEqual(res.status_code, 400)
        data = json.loads(res.data)
        self.assertFalse(data.get("success"))
        print("[PASS] Test 4: Missing input handled correctly with HTTP 400.")

if __name__ == "__main__":
    unittest.main()

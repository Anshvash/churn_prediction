import unittest
import joblib
from sklearn.ensemble import RandomForestClassifier

class TestChurnModel(unittest.TestCase):
    def test_model_loading_and_type(self):
        """Check if the saved model is a RandomForestClassifier"""
        model = joblib.load("model/churn_model.pkl")
        self.assertIsInstance(model, RandomForestClassifier)

    def test_feature_importances(self):
        """Check model has feature importances (not empty)"""
        model = joblib.load("model/churn_model.pkl")
        self.assertGreater(len(model.feature_importances_), 0)

if __name__ == "__main__":
    unittest.main()

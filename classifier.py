"""
classifier.py
The "ML Risk Classifier" component (Sec 6.1: "e.g., gradient-boosted decision
tree or lightweight neural classifier"). Implemented here with XGBoost
gradient-boosted trees, which matches the disclosure's claim language more
literally than a bagged random forest.

A single multiclass model predicts P(benign), P(injection), P(exhaustion)
for a request. The graduated risk score is risk = 1 - P(benign), which spans
BOTH content-based (injection) and resource-consumption-based (exhaustion)
threat categories, per Claim 1. The predicted subtype is also retained so
the decision engine can pick an appropriate mitigation.

Includes a `partial_retrain` method used by the Feedback Loop (Sec 6.1 /
Sec 6.2) to simulate periodic retraining on logged outcomes, including cases
the original model misses (concept drift).
"""

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from features import FEATURE_COLUMNS
import pandas as pd


class RiskClassifier:
    def __init__(self, random_state=0, n_estimators=200, max_depth=5, learning_rate=0.1):
        self.params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=random_state,
        )
        self.model = XGBClassifier(**self.params)
        self.label_encoder = LabelEncoder()
        self.classes_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        y_enc = self.label_encoder.fit_transform(y)
        self.model.fit(X, y_enc)
        self.classes_ = list(self.label_encoder.classes_)
        return self

    def predict_proba_df(self, X: pd.DataFrame) -> pd.DataFrame:
        proba = self.model.predict_proba(X)
        return pd.DataFrame(proba, columns=self.classes_, index=X.index)

    def predict(self, X: pd.DataFrame):
        pred_enc = self.model.predict(X)
        return self.label_encoder.inverse_transform(pred_enc)

    def risk_scores(self, X: pd.DataFrame):
        proba = self.predict_proba_df(X)
        risk = 1.0 - proba.get("benign", 0.0)
        subtype = proba.drop(columns=["benign"], errors="ignore").idxmax(axis=1)
        return risk, subtype, proba

    def feature_importances(self):
        return pd.Series(self.model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)

    def partial_retrain(self, X_new: pd.DataFrame, y_new: pd.Series, X_hist: pd.DataFrame, y_hist: pd.Series,
                         hist_sample_frac=0.5, random_state=1):
        """Simulate the feedback loop: retrain from scratch on newly logged
        outcomes plus a sample of historical data (to avoid catastrophic
        forgetting)."""
        hist_sample = X_hist.sample(frac=hist_sample_frac, random_state=random_state)
        y_hist_sample = y_hist.loc[hist_sample.index]
        X_combined = pd.concat([hist_sample, X_new], ignore_index=True)
        y_combined = pd.concat([y_hist_sample, y_new], ignore_index=True)
        self.fit(X_combined, y_combined)
        return self
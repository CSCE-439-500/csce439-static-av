from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.svm import SVC


@dataclass
class SklearnBaselineModel:
    """
    Train SVM on a single feature n_imports.
    """

    C: float = 1.0
    kernel: str = "rbf"
    gamma: str | float = "scale"
    random_state: int | None = None

    def __post_init__(self) -> None:
        self.clf = SVC(
            C=self.C,
            kernel=self.kernel,
            gamma=self.gamma,
            probability=True,
            random_state=self.random_state,
        )

    def fit(self, X, y) -> SklearnBaselineModel:
        self.clf.fit(X, y)
        return self

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        # SVC(probability=True) exposes predict_proba
        return self.clf.predict_proba(X)

    def model_info(self) -> dict[str, Any]:
        return {
            "name": "sklearn_svc_class_demo",
            "features": ["n_imports"],
            "classifier": "SVC",
            "params": {"C": self.C, "kernel": self.kernel, "gamma": self.gamma},
        }

import numpy as np


class SimpleLabelEncoder:
    """Minimal label encoder compatible with fit/transform/inverse_transform workflow."""

    def __init__(self):
        self.classes_ = None
        self.class_to_idx = None

    def fit(self, classes):
        unique = list(dict.fromkeys(classes))
        self.classes_ = np.array(unique)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        return self

    def transform(self, labels):
        if self.class_to_idx is None:
            raise ValueError("SimpleLabelEncoder must be fitted before calling transform().")
        return np.array([self.class_to_idx[l] for l in labels], dtype=np.int64)

    def inverse_transform(self, idx):
        if self.classes_ is None:
            raise ValueError("SimpleLabelEncoder must be fitted before calling inverse_transform().")
        idx = np.array(idx, dtype=np.int64)
        return self.classes_[idx]

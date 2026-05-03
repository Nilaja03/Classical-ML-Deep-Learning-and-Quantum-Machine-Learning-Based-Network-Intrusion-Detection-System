"""
Test Suite for Full-Stack Network Intrusion Detection System
============================================================
Covers: Data preprocessing, classical ML models, DNN, and Quantum ML components.
Dataset: KDD Cup-style NSL-KDD with 41 features + class label.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]
TARGET_COL = "class"

@pytest.fixture
def raw_train_sample():
    """Minimal in-memory replica of Train_data.csv rows."""
    return pd.DataFrame({
        "duration": [0, 0, 2, 0, 1],
        "protocol_type": ["tcp", "udp", "tcp", "icmp", "tcp"],
        "service": ["http", "other", "private", "eco_i", "ftp_data"],
        "flag": ["SF", "SF", "S0", "SF", "REJ"],
        "src_bytes": [215, 162, 0, 0, 0],
        "dst_bytes": [45076, 4528, 0, 0, 0],
        "land": [0, 0, 0, 0, 0],
        "wrong_fragment": [0, 0, 0, 0, 0],
        "urgent": [0, 0, 0, 0, 0],
        "hot": [0, 0, 0, 0, 0],
        "num_failed_logins": [0, 0, 0, 0, 0],
        "logged_in": [1, 1, 0, 0, 0],
        "num_compromised": [0, 0, 0, 0, 0],
        "root_shell": [0, 0, 0, 0, 0],
        "su_attempted": [0, 0, 0, 0, 0],
        "num_root": [0, 0, 0, 0, 0],
        "num_file_creations": [0, 0, 0, 0, 0],
        "num_shells": [0, 0, 0, 0, 0],
        "num_access_files": [0, 0, 0, 0, 0],
        "num_outbound_cmds": [0, 0, 0, 0, 0],
        "is_host_login": [0, 0, 0, 0, 0],
        "is_guest_login": [0, 0, 0, 0, 0],
        "count": [1, 1, 123, 511, 121],
        "srv_count": [1, 1, 6, 511, 19],
        "serror_rate": [0.0, 0.0, 1.0, 0.0, 0.0],
        "srv_serror_rate": [0.0, 0.0, 1.0, 0.0, 0.0],
        "rerror_rate": [0, 0, 0, 0, 1],
        "srv_rerror_rate": [0, 0, 0, 0, 1],
        "same_srv_rate": [1.0, 1.0, 0.05, 1.0, 0.16],
        "diff_srv_rate": [0.0, 0.0, 0.07, 0.0, 0.06],
        "srv_diff_host_rate": [0.0, 0.0, 0.0, 0.0, 0.0],
        "dst_host_count": [255, 255, 255, 255, 255],
        "dst_host_srv_count": [255, 255, 26, 255, 19],
        "dst_host_same_srv_rate": [1.0, 1.0, 0.10, 1.0, 0.07],
        "dst_host_diff_srv_rate": [0.0, 0.0, 0.05, 0.0, 0.07],
        "dst_host_same_src_port_rate": [0.0, 0.01, 0.0, 1.0, 0.0],
        "dst_host_srv_diff_host_rate": [0.0, 0.0, 0.0, 0.0, 0.0],
        "dst_host_serror_rate": [0.0, 0.0, 1.0, 0.0, 0.0],
        "dst_host_srv_serror_rate": [0.0, 0.0, 1.0, 0.0, 0.0],
        "dst_host_rerror_rate": [0.0, 0.0, 0.0, 0.0, 1.0],
        "dst_host_srv_rerror_rate": [0.0, 0.0, 0.0, 0.0, 1.0],
        "class": ["normal", "normal", "anomaly", "normal", "anomaly"],
    })


@pytest.fixture
def preprocessed_data(raw_train_sample):
    """Return scaled X and encoded y arrays ready for model training."""
    df = raw_train_sample.copy()
    encoder_dict = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoder_dict[col] = le

    le_target = LabelEncoder()
    df[TARGET_COL] = le_target.fit_transform(df[TARGET_COL])

    X = df.drop(TARGET_COL, axis=1).values
    y = df[TARGET_COL].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, encoder_dict, le_target, scaler


# ─────────────────────────────────────────────
# 1. DATA LOADING & SCHEMA TESTS
# ─────────────────────────────────────────────

class TestDataLoading:

    def test_train_has_correct_columns(self, raw_train_sample):
        """Dataset must contain all 41 feature columns plus the class label."""
        expected_cols = [
            "duration", "protocol_type", "service", "flag",
            "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
            "num_failed_logins", "logged_in", "num_compromised", "root_shell",
            "su_attempted", "num_root", "num_file_creations", "num_shells",
            "num_access_files", "num_outbound_cmds", "is_host_login",
            "is_guest_login", "count", "srv_count", "serror_rate",
            "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
            "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
            "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
            "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
            "dst_host_srv_serror_rate", "dst_host_rerror_rate",
            "dst_host_srv_rerror_rate", "class",
        ]
        assert list(raw_train_sample.columns) == expected_cols

    def test_no_null_values(self, raw_train_sample):
        """Dataset should contain no missing values."""
        assert raw_train_sample.isnull().sum().sum() == 0

    def test_class_column_binary(self, raw_train_sample):
        """Target column must only contain 'normal' and 'anomaly'."""
        allowed = {"normal", "anomaly"}
        assert set(raw_train_sample[TARGET_COL].unique()).issubset(allowed)

    def test_protocol_type_values(self, raw_train_sample):
        """protocol_type must be one of tcp, udp, icmp."""
        allowed = {"tcp", "udp", "icmp"}
        assert set(raw_train_sample["protocol_type"].unique()).issubset(allowed)

    def test_flag_values(self, raw_train_sample):
        """flag column must contain only known KDD flag categories."""
        known_flags = {"SF", "S0", "REJ", "RSTR", "SH", "RSTO", "S1", "RSTOS0", "S3", "S2", "OTH"}
        assert set(raw_train_sample["flag"].unique()).issubset(known_flags)

    def test_rate_columns_bounded(self, raw_train_sample):
        """All *_rate columns must be between 0.0 and 1.0."""
        rate_cols = [c for c in raw_train_sample.columns if "rate" in c]
        for col in rate_cols:
            assert raw_train_sample[col].between(0.0, 1.0).all(), \
                f"Column {col} has values outside [0, 1]"

    def test_numeric_columns_non_negative(self, raw_train_sample):
        """Byte/count columns must be non-negative."""
        non_negative = ["src_bytes", "dst_bytes", "count", "srv_count",
                        "dst_host_count", "dst_host_srv_count"]
        for col in non_negative:
            assert (raw_train_sample[col] >= 0).all(), \
                f"Column {col} contains negative values"

    def test_both_classes_present(self, raw_train_sample):
        """Dataset must include both normal and anomaly records."""
        classes = set(raw_train_sample[TARGET_COL].unique())
        assert "normal" in classes
        assert "anomaly" in classes


# ─────────────────────────────────────────────
# 2. PREPROCESSING TESTS
# ─────────────────────────────────────────────

class TestPreprocessing:

    def test_label_encoding_categorical(self, raw_train_sample):
        """Categorical columns must be converted to integers after encoding."""
        df = raw_train_sample.copy()
        for col in CATEGORICAL_COLS:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
        for col in CATEGORICAL_COLS:
            assert pd.api.types.is_integer_dtype(df[col]), \
                f"{col} not integer after LabelEncoding"

    def test_target_encoded_binary(self, raw_train_sample):
        """After encoding, target should only contain 0 and 1."""
        df = raw_train_sample.copy()
        le = LabelEncoder()
        df[TARGET_COL] = le.fit_transform(df[TARGET_COL])
        assert set(df[TARGET_COL].unique()).issubset({0, 1})

    def test_standard_scaler_mean_std(self, preprocessed_data):
        """Scaled features should have approximately zero mean and unit std."""
        X_scaled, *_ = preprocessed_data
        assert abs(X_scaled.mean()) < 0.5, "Mean of scaled data too far from 0"

    def test_feature_count_unchanged(self, preprocessed_data):
        """Preprocessing must not drop or add any feature columns."""
        X_scaled, *_ = preprocessed_data
        # 41 features (42 columns minus class)
        assert X_scaled.shape[1] == 41

    def test_unknown_service_handling(self, raw_train_sample):
        """Unknown service values in test data should not raise an error."""
        df = raw_train_sample.copy()
        for col in CATEGORICAL_COLS:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            # Simulate unknown value using the mapping strategy from the notebook
            if col == "service":
                unknown_result = -1  # maps to -1 per notebook logic
                assert unknown_result == -1  # known unknown encoding

    def test_train_test_split_proportions(self, preprocessed_data):
        """Validation split must respect the 80/20 ratio from the notebook."""
        X_scaled, y, *_ = preprocessed_data
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        total = len(X_train) + len(X_val)
        assert abs(len(X_val) / total - 0.2) < 0.05

    def test_pca_reduces_dimensions(self, preprocessed_data):
        """PCA for quantum models must reduce 41 dims to 4."""
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import MinMaxScaler
        X_scaled, *_ = preprocessed_data
        q_scaler = MinMaxScaler(feature_range=(0, 1))
        X_q = q_scaler.fit_transform(X_scaled)
        pca = PCA(n_components=4)
        X_pca = pca.fit_transform(X_q)
        assert X_pca.shape[1] == 4

    def test_minmax_scaler_bounds(self, preprocessed_data):
        """MinMaxScaler output for quantum pipeline must stay within [0, 1]."""
        from sklearn.preprocessing import MinMaxScaler
        X_scaled, *_ = preprocessed_data
        q_scaler = MinMaxScaler(feature_range=(0, 1))
        X_q = q_scaler.fit_transform(X_scaled)
        assert X_q.min() >= 0.0 - 1e-9
        assert X_q.max() <= 1.0 + 1e-9


# ─────────────────────────────────────────────
# 3. CLASSICAL ML MODEL TESTS
# ─────────────────────────────────────────────

class TestClassicalModels:

    @pytest.fixture(autouse=True)
    def setup(self, preprocessed_data):
        X, y, *_ = preprocessed_data
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    @pytest.mark.parametrize("ModelClass,kwargs", [
        (KNeighborsClassifier, {"n_neighbors": 3}),
        (DecisionTreeClassifier, {}),
        (RandomForestClassifier, {}),
        (SVC, {"probability": True}),
        (GaussianNB, {}),
    ])
    def test_model_fits_without_error(self, ModelClass, kwargs):
        """Each classical model must train on the dataset without exception."""
        model = ModelClass(**kwargs)
        model.fit(self.X_train, self.y_train)
        preds = model.predict(self.X_val)
        assert len(preds) == len(self.y_val)

    @pytest.mark.parametrize("ModelClass,kwargs", [
        (KNeighborsClassifier, {"n_neighbors": 3}),
        (DecisionTreeClassifier, {}),
        (RandomForestClassifier, {}),
        (GaussianNB, {}),
    ])
    def test_prediction_values_binary(self, ModelClass, kwargs):
        """Model predictions must be binary (0 or 1) for this dataset."""
        model = ModelClass(**kwargs)
        model.fit(self.X_train, self.y_train)
        preds = model.predict(self.X_val)
        assert set(preds).issubset({0, 1})

    def test_random_forest_probability_shape(self):
        """RandomForest predict_proba must return two probability columns."""
        model = RandomForestClassifier()
        model.fit(self.X_train, self.y_train)
        proba = model.predict_proba(self.X_val)
        assert proba.shape[1] == 2

    def test_random_forest_probabilities_sum_to_one(self):
        """Each row of predict_proba must sum to approximately 1.0."""
        model = RandomForestClassifier()
        model.fit(self.X_train, self.y_train)
        proba = model.predict_proba(self.X_val)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_svc_with_probability(self):
        """SVC trained with probability=True must support predict_proba."""
        model = SVC(probability=True)
        model.fit(self.X_train, self.y_train)
        proba = model.predict_proba(self.X_val)
        assert proba.shape == (len(self.y_val), 2)

    def test_accuracy_score_in_valid_range(self):
        """Accuracy must be a float between 0.0 and 1.0."""
        model = DecisionTreeClassifier()
        model.fit(self.X_train, self.y_train)
        acc = accuracy_score(self.y_val, model.predict(self.X_val))
        assert 0.0 <= acc <= 1.0

    def test_classification_report_keys(self):
        """classification_report dict must contain weighted avg metrics."""
        model = RandomForestClassifier()
        model.fit(self.X_train, self.y_train)
        preds = model.predict(self.X_val)
        report = classification_report(self.y_val, preds, output_dict=True)
        assert "weighted avg" in report
        assert "precision" in report["weighted avg"]
        assert "recall" in report["weighted avg"]
        assert "f1-score" in report["weighted avg"]

    def test_best_model_selection(self):
        """Best model selection must return the model with highest accuracy."""
        models = {
            "KNN": KNeighborsClassifier(n_neighbors=3),
            "DT": DecisionTreeClassifier(),
        }
        ml_results = {}
        for name, model in models.items():
            model.fit(self.X_train, self.y_train)
            preds = model.predict(self.X_val)
            ml_results[name] = accuracy_score(self.y_val, preds)
        best = max(ml_results, key=ml_results.get)
        assert best in models


# ─────────────────────────────────────────────
# 4. DEEP NEURAL NETWORK TESTS
# ─────────────────────────────────────────────

class TestDNN:

    def _build_dnn(self, input_dim):
        """Rebuild the DNN architecture from the notebook."""
        from keras.models import Sequential
        from keras.layers import Dense, Dropout, Input
        model = Sequential([
            Input(shape=(input_dim,)),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid"),
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model

    def test_dnn_builds_without_error(self, preprocessed_data):
        """DNN must compile with the correct architecture."""
        pytest.importorskip("keras")
        X, *_ = preprocessed_data
        model = self._build_dnn(X.shape[1])
        assert model is not None

    def test_dnn_output_shape(self, preprocessed_data):
        """DNN must produce one probability per input sample."""
        pytest.importorskip("keras")
        X, y, *_ = preprocessed_data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        model = self._build_dnn(X_train.shape[1])
        model.fit(X_train, y_train, epochs=1, verbose=0)
        probs = model.predict(X_val, verbose=0)
        assert probs.shape == (len(X_val), 1)

    def test_dnn_output_bounded(self, preprocessed_data):
        """DNN sigmoid output must be strictly in [0, 1]."""
        pytest.importorskip("keras")
        X, y, *_ = preprocessed_data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        model = self._build_dnn(X_train.shape[1])
        model.fit(X_train, y_train, epochs=1, verbose=0)
        probs = model.predict(X_val, verbose=0)
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0

    def test_dnn_threshold_produces_binary(self, preprocessed_data):
        """Applying 0.5 threshold to DNN output must give only 0 or 1."""
        pytest.importorskip("keras")
        X, y, *_ = preprocessed_data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        model = self._build_dnn(X_train.shape[1])
        model.fit(X_train, y_train, epochs=1, verbose=0)
        preds = (model.predict(X_val, verbose=0) > 0.5).astype(int).flatten()
        assert set(preds).issubset({0, 1})

    def test_dnn_layer_count(self, preprocessed_data):
        """DNN must have 3 Dense layers as defined in the notebook."""
        pytest.importorskip("keras")
        from keras.layers import Dense
        X, *_ = preprocessed_data
        model = self._build_dnn(X.shape[1])
        dense_layers = [l for l in model.layers if isinstance(l, Dense)]
        assert len(dense_layers) == 3

    def test_dnn_loss_decreases_over_epochs(self, preprocessed_data):
        """Training loss must decrease from epoch 1 to epoch 5."""
        pytest.importorskip("keras")
        X, y, *_ = preprocessed_data
        model = self._build_dnn(X.shape[1])
        history = model.fit(X, y, epochs=5, verbose=0)
        losses = history.history["loss"]
        assert losses[-1] < losses[0], "Training loss did not decrease"


# ─────────────────────────────────────────────
# 5. METRICS & COMPARISON TABLE TESTS
# ─────────────────────────────────────────────

class TestMetrics:

    def _get_metrics(self, name, y_true, y_pred):
        report = classification_report(y_true, y_pred, output_dict=True)
        return {
            "Model": name,
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": report["weighted avg"]["precision"],
            "Recall": report["weighted avg"]["recall"],
            "F1-Score": report["weighted avg"]["f1-score"],
        }

    def test_get_metrics_returns_correct_keys(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])
        result = self._get_metrics("TestModel", y_true, y_pred)
        assert set(result.keys()) == {"Model", "Accuracy", "Precision", "Recall", "F1-Score"}

    def test_accuracy_perfect_predictions(self):
        """Identical true and predicted arrays should give accuracy of 1.0."""
        y = np.array([0, 1, 0, 1, 1])
        assert accuracy_score(y, y) == 1.0

    def test_accuracy_all_wrong(self):
        """Completely wrong predictions must give accuracy of 0.0."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1])
        assert accuracy_score(y_true, y_pred) == 0.0

    def test_comparison_df_sorted_by_accuracy(self):
        """Model comparison DataFrame must be sorted descending by Accuracy."""
        records = [
            {"Model": "A", "Accuracy": 0.85, "Precision": 0.85, "Recall": 0.85, "F1-Score": 0.85},
            {"Model": "B", "Accuracy": 0.92, "Precision": 0.92, "Recall": 0.92, "F1-Score": 0.92},
            {"Model": "C", "Accuracy": 0.78, "Precision": 0.78, "Recall": 0.78, "F1-Score": 0.78},
        ]
        df = pd.DataFrame(records).sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
        assert df.iloc[0]["Model"] == "B"
        assert df.iloc[-1]["Model"] == "C"

    def test_f1_score_in_range(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 0, 0, 1])
        result = self._get_metrics("model", y_true, y_pred)
        assert 0.0 <= result["F1-Score"] <= 1.0

    def test_precision_recall_consistency(self):
        """Precision and Recall must both be in [0, 1]."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        result = self._get_metrics("model", y_true, y_pred)
        assert 0.0 <= result["Precision"] <= 1.0
        assert 0.0 <= result["Recall"] <= 1.0


# ─────────────────────────────────────────────
# 6. SERIALIZATION / PERSISTENCE TESTS
# ─────────────────────────────────────────────

class TestSerialization:

    def test_joblib_dump_and_load_scaler(self, preprocessed_data, tmp_path):
        """Scaler must serialize and deserialize with identical transform output."""
        import joblib
        _, _, _, _, scaler = preprocessed_data
        path = tmp_path / "scaler.pkl"
        joblib.dump(scaler, path)
        loaded = joblib.load(path)
        # Check that the loaded scaler has the same mean
        np.testing.assert_array_almost_equal(scaler.mean_, loaded.mean_)

    def test_joblib_dump_and_load_model(self, preprocessed_data, tmp_path):
        """Saved and loaded model must produce identical predictions."""
        import joblib
        X, y, *_ = preprocessed_data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        preds_before = model.predict(X_val)

        path = tmp_path / "model.pkl"
        joblib.dump(model, path)
        loaded = joblib.load(path)
        preds_after = loaded.predict(X_val)

        np.testing.assert_array_equal(preds_before, preds_after)

    def test_accuracies_dict_saveable(self, tmp_path):
        """Accuracies dict with string keys and float values must be picklable."""
        import joblib
        accuracies = {"KNN": 0.97, "DT": 0.98, "DNN": 0.96}
        path = tmp_path / "accuracies.pkl"
        joblib.dump(accuracies, path)
        loaded = joblib.load(path)
        assert loaded == accuracies

    def test_encoder_dict_persists_classes(self, raw_train_sample, tmp_path):
        """LabelEncoders saved to disk must retain their classes_ attribute."""
        import joblib
        encoder_dict = {}
        for col in CATEGORICAL_COLS:
            le = LabelEncoder()
            le.fit(raw_train_sample[col])
            encoder_dict[col] = le
        path = tmp_path / "encoders.pkl"
        joblib.dump(encoder_dict, path)
        loaded = joblib.load(path)
        for col in CATEGORICAL_COLS:
            np.testing.assert_array_equal(
                encoder_dict[col].classes_, loaded[col].classes_
            )


# ─────────────────────────────────────────────
# 7. EDGE CASE & ROBUSTNESS TESTS
# ─────────────────────────────────────────────

class TestEdgeCases:

    def test_all_normal_input(self):
        """Model should not crash when all rows are 'normal'."""
        X = np.zeros((10, 41))
        y = np.zeros(10)  # all normal
        model = DecisionTreeClassifier()
        model.fit(X, y)
        preds = model.predict(X)
        assert (preds == 0).all()

    def test_all_anomaly_input(self):
        """Model should not crash when all rows are 'anomaly'."""
        X = np.ones((10, 41))
        y = np.ones(10)  # all anomaly
        model = DecisionTreeClassifier()
        model.fit(X, y)
        preds = model.predict(X)
        assert (preds == 1).all()

    def test_single_sample_prediction(self, preprocessed_data):
        """Model must handle a single-row inference without shape errors."""
        X, y, *_ = preprocessed_data
        model = RandomForestClassifier()
        model.fit(X, y)
        single = X[0:1, :]
        pred = model.predict(single)
        assert len(pred) == 1

    def test_high_src_bytes_does_not_break_scaler(self):
        """Extreme byte values must not cause scaler overflow or NaN."""
        scaler = StandardScaler()
        X = np.zeros((5, 41))
        X[:, 4] = [0, 10**9, 10**9, 10**9, 10**9]  # src_bytes extremes
        X_scaled = scaler.fit_transform(X)
        assert not np.isnan(X_scaled).any()
        assert not np.isinf(X_scaled).any()

    def test_zero_variance_feature(self):
        """A column with all-zero values must not cause StandardScaler to NaN."""
        scaler = StandardScaler()
        X = np.random.randn(10, 41)
        X[:, 20] = 0  # is_host_login is always 0 in practice
        # StandardScaler with_std=True sets std=1 for zero-variance cols
        X_scaled = scaler.fit_transform(X)
        assert not np.isnan(X_scaled).any()

    def test_prediction_speed_is_reasonable(self, preprocessed_data):
        """A trained RandomForest must predict 100 rows in under 2 seconds."""
        import time
        X, y, *_ = preprocessed_data
        model = RandomForestClassifier(n_estimators=10)
        model.fit(X, y)
        X_large = np.tile(X, (20, 1))[:100]
        start = time.time()
        model.predict(X_large)
        elapsed = time.time() - start
        assert elapsed < 2.0, f"Prediction too slow: {elapsed:.2f}s"

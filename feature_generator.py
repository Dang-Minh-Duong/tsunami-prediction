import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.utils.validation import check_is_fitted


class PhysicsFeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.new_features_ = [
            "log_depth",
            "log_sig",
            "energy_release",
            "log_energy",
            "mag_depth_ratio",
            "mag_log_depth",
            "is_shallow",
            "is_major",
            "tsunami_potential",
            "x_coord",
            "y_coord",
            "z_coord",
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        X_new = X.copy()
        magnitude = X_new["magnitude"].values
        depth = X_new["depth"].values

        X_new["log_depth"] = np.log1p(depth)
        X_new["log_sig"] = np.log1p(X_new["sig"].values)
        X_new["energy_release"] = 10 ** (1.5 * magnitude)
        X_new["log_energy"] = 1.5 * magnitude
        X_new["mag_depth_ratio"] = magnitude / (depth + 1.0)
        X_new["tsunami_potential"] = (magnitude**2) / np.sqrt(depth + 1e-6)
        X_new["is_shallow"] = (depth < 70).astype(int)
        X_new["is_major"] = (magnitude >= 7.5).astype(int)
        X_new["mag_log_depth"] = magnitude * X_new["log_depth"]

        lat_rad = np.radians(X_new["latitude"].values)
        lon_rad = np.radians(X_new["longitude"].values)
        X_new["x_coord"] = np.cos(lat_rad) * np.cos(lon_rad)
        X_new["y_coord"] = np.cos(lat_rad) * np.sin(lon_rad)
        X_new["z_coord"] = np.sin(lat_rad)
        return X_new

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array(self.new_features_, dtype=object)
        return np.array(list(input_features) + self.new_features_, dtype=object)


class AdaptiveSphericalRBF(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=5, random_state=2):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X, y=None):
        coordinates = X[["x_coord", "y_coord", "z_coord"]].values
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
        ).fit(coordinates)

        centers = self.kmeans_.cluster_centers_
        self.cluster_centers_ = centers / np.linalg.norm(
            centers, axis=1, keepdims=True
        )
        labels = self.kmeans_.labels_
        self.cluster_gammas_ = np.zeros(self.n_clusters)

        for cluster in range(self.n_clusters):
            mask = labels == cluster
            if np.sum(mask) > 1:
                cluster_points = coordinates[mask]
                dot = cluster_points @ self.cluster_centers_[cluster]
                angles = np.arccos(np.clip(dot, -1.0, 1.0))
                sigma_sq = np.mean(angles**2)
                self.cluster_gammas_[cluster] = 1.0 / (2 * sigma_sq + 1e-9)
            else:
                self.cluster_gammas_[cluster] = 50.0
        return self

    def transform(self, X):
        check_is_fitted(self, ["cluster_centers_", "cluster_gammas_"])
        coordinates = X[["x_coord", "y_coord", "z_coord"]].values
        dot = coordinates @ self.cluster_centers_.T
        angles_sq = np.arccos(np.clip(dot, -1.0, 1.0)) ** 2
        rbf_features = np.exp(-angles_sq * self.cluster_gammas_)

        if isinstance(X, pd.DataFrame):
            columns = [f"geo_rbf_{i}" for i in range(self.n_clusters)]
            rbf_df = pd.DataFrame(rbf_features, columns=columns, index=X.index)
            return pd.concat([X, rbf_df], axis=1)
        return np.hstack([X, rbf_features])

    def get_feature_names_out(self, input_features=None):
        rbf_names = [f"geo_rbf_{i}" for i in range(self.n_clusters)]
        if input_features is None:
            return rbf_names
        return list(input_features) + rbf_names


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        valid_columns = [
            column for column in self.feature_names if column in X.columns
        ]
        return X[valid_columns]

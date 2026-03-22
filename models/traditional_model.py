"""
Traditional Machine Learning models for plankton classification.
Includes SVM and Random Forest classifiers.
"""

import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report


class TraditionalMLClassifier:
    """
    Wrapper for traditional ML classifiers (SVM, Random Forest).
    """

    def __init__(
        self,
        classifier_type: str = 'svm',
        config_dict: dict = None
    ):
        self.classifier_type = classifier_type
        self.config = config_dict or {}
        self.model = None
        self.classes_ = None
        self.feature_size = None

    def create_model(self):
        """Create the classifier based on type."""
        if self.classifier_type == 'svm':
            self.model = SVC(
                kernel=self.config.get('svm_kernel', 'rbf'),
                C=self.config.get('svm_C', 10),
                gamma=self.config.get('svm_gamma', 'scale'),
                probability=True,
                class_weight='balanced',  # Handle imbalance
                random_state=42,
                verbose=True,
                cache_size=1000  # Use more memory for faster training
            )
        elif self.classifier_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=self.config.get('rf_n_estimators', 300),
                max_depth=self.config.get('rf_max_depth', None),
                min_samples_split=self.config.get('rf_min_samples_split', 2),
                class_weight='balanced',
                n_jobs=self.config.get('n_jobs', -1),
                random_state=42,
                verbose=1
            )
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")

        return self.model

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the classifier."""
        if self.model is None:
            self.create_model()

        self.feature_size = X.shape[1]
        self.classes_ = np.unique(y)

        print(f"\nTraining {self.classifier_type.upper()}...")
        print(f"  Training samples: {X.shape[0]:,}")
        print(f"  Feature size: {X.shape[1]}")
        print(f"  Number of classes: {len(self.classes_)}")

        self.model.fit(X, y)

        # Training accuracy
        train_pred = self.model.predict(X)
        train_acc = accuracy_score(y, train_pred)
        print(f"  Training accuracy: {train_acc:.4f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(X)

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        class_names: Optional[List[str]] = None
    ) -> Dict:
        """Evaluate the model."""
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)

        accuracy = accuracy_score(y, predictions)

        # Top-k accuracy
        top_k_accs = {}
        for k in [1, 3, 5]:
            if k == 1:
                top_k_accs[f'top_{k}_accuracy'] = accuracy
            else:
                top_k_correct = 0
                top_k_indices = np.argsort(probabilities, axis=1)[:, -k:]
                for i, true_label in enumerate(y):
                    if true_label in top_k_indices[i]:
                        top_k_correct += 1
                top_k_accs[f'top_{k}_accuracy'] = top_k_correct / len(y)

        results = {
            'accuracy': accuracy,
            **top_k_accs,
            'predictions': predictions,
            'probabilities': probabilities,
            'classification_report': classification_report(
                y, predictions,
                target_names=class_names,
                output_dict=True
            ) if class_names else None
        }

        return results

    def save(self, path: Path):
        """Save the model to disk."""
        joblib.dump({
            'model': self.model,
            'classifier_type': self.classifier_type,
            'config': self.config,
            'classes_': self.classes_,
            'feature_size': self.feature_size
        }, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Path) -> 'TraditionalMLClassifier':
        """Load a model from disk."""
        data = joblib.load(path)

        instance = cls(
            classifier_type=data['classifier_type'],
            config_dict=data['config']
        )
        instance.model = data['model']
        instance.classes_ = data['classes_']
        instance.feature_size = data['feature_size']

        return instance

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance (only for Random Forest)."""
        if self.classifier_type == 'random_forest' and self.model is not None:
            return self.model.feature_importances_
        return None


def create_svm(config_dict: dict = None) -> TraditionalMLClassifier:
    """Create an SVM classifier."""
    from config import TRADITIONAL_ML_CONFIG
    cfg = config_dict or TRADITIONAL_ML_CONFIG
    return TraditionalMLClassifier(classifier_type='svm', config_dict=cfg)


def create_random_forest(config_dict: dict = None) -> TraditionalMLClassifier:
    """Create a Random Forest classifier."""
    from config import TRADITIONAL_ML_CONFIG
    cfg = config_dict or TRADITIONAL_ML_CONFIG
    return TraditionalMLClassifier(classifier_type='random_forest', config_dict=cfg)


class EnsembleClassifier:
    """
    Ensemble of multiple traditional ML classifiers.
    Uses soft voting (probability averaging).
    """

    def __init__(self, classifiers: List[TraditionalMLClassifier]):
        self.classifiers = classifiers
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train all classifiers."""
        self.classes_ = np.unique(y)

        for clf in self.classifiers:
            clf.fit(X, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using averaged probabilities."""
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Average predicted probabilities from all classifiers."""
        probas = np.zeros((X.shape[0], len(self.classes_)))

        for clf in self.classifiers:
            probas += clf.predict_proba(X)

        probas /= len(self.classifiers)
        return probas

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        class_names: Optional[List[str]] = None
    ) -> Dict:
        """Evaluate the ensemble."""
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)

        accuracy = accuracy_score(y, predictions)

        # Top-k accuracy
        top_k_accs = {}
        for k in [1, 3, 5]:
            if k == 1:
                top_k_accs[f'top_{k}_accuracy'] = accuracy
            else:
                top_k_correct = 0
                top_k_indices = np.argsort(probabilities, axis=1)[:, -k:]
                for i, true_label in enumerate(y):
                    if true_label in top_k_indices[i]:
                        top_k_correct += 1
                top_k_accs[f'top_{k}_accuracy'] = top_k_correct / len(y)

        return {
            'accuracy': accuracy,
            **top_k_accs,
            'predictions': predictions,
            'probabilities': probabilities
        }


def hyperparameter_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    classifier_type: str = 'svm',
    cv: int = 3
) -> Dict:
    """
    Perform hyperparameter search for traditional ML classifiers.

    Note: This can be very slow for large datasets.
    """
    print(f"\nPerforming hyperparameter search for {classifier_type}...")

    if classifier_type == 'svm':
        model = SVC(probability=True, class_weight='balanced', random_state=42)
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.1],
            'kernel': ['rbf', 'poly']
        }
    elif classifier_type == 'random_forest':
        model = RandomForestClassifier(
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")

    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X_train, y_train)

    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

    return {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'best_estimator': grid_search.best_estimator_,
        'cv_results': grid_search.cv_results_
    }


if __name__ == "__main__":
    # Test model creation
    print("Testing Traditional ML model creation...")

    import config

    # Create SVM
    svm = create_svm(config.TRADITIONAL_ML_CONFIG)
    svm.create_model()
    print(f"SVM model: {svm.model}")

    # Create Random Forest
    rf = create_random_forest(config.TRADITIONAL_ML_CONFIG)
    rf.create_model()
    print(f"Random Forest model: {rf.model}")

    # Test with dummy data
    X_dummy = np.random.randn(100, 500)
    y_dummy = np.random.randint(0, 10, 100)

    print("\nTesting SVM fit...")
    svm.fit(X_dummy, y_dummy)

    print("\nTesting Random Forest fit...")
    rf.fit(X_dummy, y_dummy)

    print("\nModel tests completed!")

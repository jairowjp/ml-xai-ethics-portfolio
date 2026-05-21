"""
model_training.py
=================
Módulo de entrenamiento y evaluación de modelos supervisados.
Proyecto: MIAR0525 — Semana 4: Ética, Sesgo y XAI
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score,
    f1_score, precision_score, recall_score,
    ConfusionMatrixDisplay, RocCurveDisplay
)
from sklearn.model_selection import cross_val_score, StratifiedKFold

SEED = 42
COLORS = ['#8B1A2F', '#C0392B', '#E74C3C', '#5D6D7E', '#2C3E50']


# ─────────────────────────────────────────────────────────────────────────────
# Random Forest
# ─────────────────────────────────────────────────────────────────────────────

def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame,
                        n_estimators: int = 200,
                        max_depth: int = 10,
                        min_samples_leaf: int = 5) -> tuple:
    """
    Entrena un Random Forest con parámetros optimizados para Adult Income.

    Parameters
    ----------
    X_train, y_train : datos de entrenamiento
    X_test           : datos de prueba para predicción
    n_estimators     : número de árboles (default 200)
    max_depth        : profundidad máxima (default 10)
    min_samples_leaf : mínimo de muestras por hoja (default 5)

    Returns
    -------
    model    : RandomForestClassifier ajustado
    y_pred   : predicciones en X_test
    y_proba  : probabilidades en X_test
    metrics  : dict con métricas de evaluación
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight='balanced',
        random_state=SEED,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = _compute_metrics(y_test=None, y_pred=y_pred, y_proba=y_proba)
    return model, y_pred, y_proba, metrics


def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series,
                               X_test: pd.DataFrame,
                               C: float = 1.0,
                               max_iter: int = 1000) -> tuple:
    """
    Entrena una Regresión Logística.

    Returns
    -------
    model, y_pred, y_proba, metrics
    """
    model = LogisticRegression(C=C, max_iter=max_iter,
                               class_weight='balanced',
                               random_state=SEED, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = _compute_metrics(y_test=None, y_pred=y_pred, y_proba=y_proba)
    return model, y_pred, y_proba, metrics


def train_decision_tree(X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame,
                        max_depth: int = 8) -> tuple:
    """
    Entrena un Árbol de Decisión (útil para visualización explícita).

    Returns
    -------
    model, y_pred, y_proba, metrics
    """
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=SEED
    )
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = _compute_metrics(y_test=None, y_pred=y_pred, y_proba=y_proba)
    return model, y_pred, y_proba, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Evaluación
# ─────────────────────────────────────────────────────────────────────────────

def _compute_metrics(y_test, y_pred, y_proba=None) -> dict:
    """Calcula métricas estándar de clasificación binaria."""
    metrics = {
        'accuracy':  accuracy_score(y_test, y_pred) if y_test is not None else None,
        'f1':        f1_score(y_test, y_pred, zero_division=0) if y_test is not None else None,
        'precision': precision_score(y_test, y_pred, zero_division=0) if y_test is not None else None,
        'recall':    recall_score(y_test, y_pred, zero_division=0) if y_test is not None else None,
    }
    if y_proba is not None and y_test is not None:
        metrics['auc_roc'] = roc_auc_score(y_test, y_proba)
    return metrics


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series,
                   model_name: str = "Modelo",
                   save_path: str = None) -> dict:
    """
    Evaluación completa con métricas, matriz de confusión y curva ROC.

    Parameters
    ----------
    model      : modelo entrenado con predict y predict_proba
    X_test     : features de test
    y_test     : etiquetas reales
    model_name : nombre para títulos de figuras
    save_path  : si se provee, guarda la figura

    Returns
    -------
    dict con accuracy, f1, precision, recall, auc_roc
    """
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n{'='*55}")
    print(f"  {model_name} — Evaluación")
    print(f"{'='*55}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  AUC-ROC  : {auc:.4f}")
    print(f"{'='*55}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                 target_names=['≤50K', '>50K']))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f'{model_name} — Evaluación del Modelo', fontweight='bold', fontsize=13)

    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=['≤50K', '>50K'],
        cmap='Reds', colorbar=False, ax=axes[0]
    )
    axes[0].set_title('Matriz de Confusión')

    RocCurveDisplay.from_predictions(y_test, y_proba, ax=axes[1])
    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random (AUC=0.50)')
    axes[1].set_title(f'Curva ROC (AUC={auc:.3f})')
    axes[1].legend(loc='lower right')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    return {
        'accuracy': acc, 'f1': f1, 'auc_roc': auc,
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'y_pred': y_pred, 'y_proba': y_proba
    }


def cross_validate_model(model, X: pd.DataFrame, y: pd.Series,
                          cv: int = 5, scoring: str = 'f1') -> dict:
    """
    Validación cruzada estratificada.

    Returns
    -------
    dict con media, desviación estándar y scores por fold
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=SEED)
    scores = cross_val_score(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)

    print(f"\nValidación Cruzada ({cv}-fold, scoring={scoring}):")
    print(f"  Media   : {scores.mean():.4f}")
    print(f"  Std Dev : {scores.std():.4f}")
    print(f"  Por fold: {[f'{s:.4f}' for s in scores]}")

    return {'mean': scores.mean(), 'std': scores.std(), 'scores': scores}


def plot_logistic_coefficients(model: LogisticRegression,
                                feature_names: list,
                                top_n: int = 10,
                                save_path: str = None) -> pd.Series:
    """
    Visualiza coeficientes de regresión logística (interpretabilidad directa).

    Parameters
    ----------
    model         : LogisticRegression ajustado
    feature_names : nombres de las variables
    top_n         : número de coeficientes a mostrar
    save_path     : ruta para guardar la figura

    Returns
    -------
    pd.Series con coeficientes ordenados por valor absoluto
    """
    coef = pd.Series(model.coef_[0], index=feature_names).sort_values(key=abs, ascending=False)
    top_coef = coef.head(top_n)

    colors_bar = ['#8B1A2F' if v > 0 else '#5D6D7E' for v in top_coef]

    fig, ax = plt.subplots(figsize=(10, 5))
    top_coef.plot(kind='bar', ax=ax, color=colors_bar, edgecolor='white')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title('Coeficientes Regresión Logística\n(rojo=positivo → >50K | gris=negativo → ≤50K)',
                  fontweight='bold')
    ax.set_xlabel('Variable')
    ax.set_ylabel('Coeficiente')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    return coef

"""
xai_utils.py
============
Utilidades para técnicas de Explicabilidad (XAI):
SHAP, LIME y Permutation Feature Importance.

MIAR0525 · Semana 4 · Ética, Sesgo y Calidad en ML
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import lime
import lime.lime_tabular
from sklearn.inspection import permutation_importance
from typing import List, Optional

COLORS = ['#8B1A2F', '#C0392B', '#A9CCE3', '#5D6D7E']
SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# SHAP
# ─────────────────────────────────────────────────────────────────────────────

def compute_shap_values(model, X_test: pd.DataFrame) -> tuple:
    """
    Calcula valores SHAP usando TreeExplainer.

    Parameters
    ----------
    model : sklearn estimator
        Modelo entrenado (Random Forest, Decision Tree, etc.)
    X_test : pd.DataFrame
        Conjunto de test.

    Returns
    -------
    tuple
        (explainer, shap_values_pos, shap_importance)
        - explainer: shap.TreeExplainer
        - shap_values_pos: np.ndarray con valores SHAP de la clase positiva
        - shap_importance: pd.Series con importancia media |SHAP|
    """
    print("⏳ Calculando valores SHAP...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Clase positiva (>50K)
    shap_pos = shap_values[1] if isinstance(shap_values, list) else shap_values

    mean_abs = np.abs(shap_pos).mean(axis=0)
    shap_importance = pd.Series(mean_abs, index=X_test.columns).sort_values(ascending=False)

    print(f"✅ SHAP calculado. Shape: {shap_pos.shape}")
    return explainer, shap_pos, shap_importance


def plot_shap_global(shap_importance: pd.Series, save_path: Optional[str] = None):
    """Gráfico de importancia global SHAP (bar horizontal)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    shap_importance.plot(kind='barh', ax=ax, color=COLORS[0], edgecolor='white')
    ax.set_title('SHAP: Importancia Global de Features\n(mean |SHAP value|)',
                 fontweight='bold', fontsize=13)
    ax.set_xlabel('mean(|SHAP value|)')
    ax.invert_yaxis()
    for i, (feat, val) in enumerate(shap_importance.items()):
        ax.text(val + 0.001, i, f'{val:.4f}', va='center', fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()


def plot_shap_waterfall(explainer, shap_pos: np.ndarray, X_test: pd.DataFrame,
                         idx: int, features: List[str], save_path: Optional[str] = None):
    """Waterfall plot SHAP para una instancia específica."""
    base_val = (explainer.expected_value[1]
                if isinstance(explainer.expected_value, list)
                else explainer.expected_value)

    shap_exp = shap.Explanation(
        values=shap_pos[idx],
        base_values=base_val,
        data=X_test.iloc[idx].values,
        feature_names=features
    )
    plt.figure(figsize=(9, 5))
    shap.waterfall_plot(shap_exp, show=False, max_display=9)
    plt.title(f'SHAP Waterfall — Instancia #{idx}', fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()


def run_shap_analysis(model, X_test: pd.DataFrame, features: List[str],
                       instance_ids: List[int] = None) -> tuple:
    """
    Pipeline completo de análisis SHAP.

    Returns
    -------
    tuple : (explainer, shap_pos, shap_importance)
    """
    explainer, shap_pos, shap_importance = compute_shap_values(model, X_test)
    plot_shap_global(shap_importance)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_pos, X_test, show=False, max_display=9)
    plt.title('SHAP Beeswarm: Dirección e Intensidad', fontweight='bold')
    plt.tight_layout()
    plt.show()

    if instance_ids is None:
        instance_ids = [0]
    for idx in instance_ids:
        plot_shap_waterfall(explainer, shap_pos, X_test, idx, features)

    return explainer, shap_pos, shap_importance


# ─────────────────────────────────────────────────────────────────────────────
# LIME
# ─────────────────────────────────────────────────────────────────────────────

def create_lime_explainer(X_train: np.ndarray, features: List[str],
                           class_names: List[str] = None) -> lime.lime_tabular.LimeTabularExplainer:
    """Inicializa un LIME TabularExplainer."""
    if class_names is None:
        class_names = ['≤50K', '>50K']
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=features,
        class_names=class_names,
        mode='classification',
        discretize_continuous=True,
        random_state=SEED
    )
    print("✅ LIME Explainer inicializado")
    return explainer


def explain_instance_lime(lime_explainer, X_test: np.ndarray, predict_fn,
                            idx: int, num_features: int = 9,
                            save_path: Optional[str] = None):
    """
    Genera y visualiza una explicación LIME para la instancia dada.

    Returns
    -------
    lime.Explanation
    """
    exp = lime_explainer.explain_instance(
        data_row=X_test[idx],
        predict_fn=predict_fn,
        num_features=num_features,
        num_samples=3000
    )
    fig = exp.as_pyplot_figure(label=1)
    fig.set_size_inches(10, 5)
    plt.title(f'LIME — Explicación Local: Instancia #{idx}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()
    return exp


def run_lime_analysis(model, X_train: np.ndarray, X_test: np.ndarray,
                       features: List[str], instance_ids: List[int] = None):
    """Pipeline completo de análisis LIME."""
    lime_exp = create_lime_explainer(X_train, features)
    if instance_ids is None:
        instance_ids = [0]
    explanations = []
    for idx in instance_ids:
        exp = explain_instance_lime(lime_exp, X_test, model.predict_proba, idx)
        explanations.append(exp)
    return lime_exp, explanations


# ─────────────────────────────────────────────────────────────────────────────
# Permutation Feature Importance
# ─────────────────────────────────────────────────────────────────────────────

def compute_permutation_importance(model, X_test: pd.DataFrame, y_test: pd.Series,
                                    features: List[str], n_repeats: int = 20,
                                    scoring: str = 'f1',
                                    save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Calcula y visualiza Permutation Feature Importance.

    Returns
    -------
    pd.DataFrame
        DataFrame con 'Feature', 'PFI_mean', 'PFI_std'
    """
    print(f"⏳ Calculando PFI (n_repeats={n_repeats}, scoring={scoring})...")
    perm = permutation_importance(model, X_test, y_test,
                                   n_repeats=n_repeats, scoring=scoring,
                                   random_state=SEED, n_jobs=-1)

    pfi_df = pd.DataFrame({
        'Feature':  features,
        'PFI_mean': perm.importances_mean,
        'PFI_std':  perm.importances_std
    }).sort_values('PFI_mean', ascending=False).reset_index(drop=True)

    # Visualización
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [COLORS[0] if v > 0.01 else COLORS[2] for v in pfi_df['PFI_mean']]
    ax.barh(pfi_df['Feature'], pfi_df['PFI_mean'], xerr=pfi_df['PFI_std'],
            color=colors, edgecolor='white',
            error_kw=dict(ecolor='gray', capsize=4, lw=1.5))
    ax.set_title(f'Permutation Feature Importance ({scoring.upper()})',
                 fontweight='bold', fontsize=13)
    ax.set_xlabel(f'Caída en {scoring.upper()} al permutar la feature')
    ax.invert_yaxis()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()

    print("✅ PFI calculado:")
    print(pfi_df.round(4).to_string(index=False))
    return pfi_df


# ─────────────────────────────────────────────────────────────────────────────
# Comparativa XAI
# ─────────────────────────────────────────────────────────────────────────────

def compare_xai_methods(shap_importance: pd.Series,
                          pfi_df: pd.DataFrame,
                          rf_importance: pd.Series,
                          save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Genera heatmap comparativo de los tres métodos XAI.

    Returns
    -------
    pd.DataFrame
        DataFrame normalizado con concordancia entre métodos.
    """
    import seaborn as sns

    ranking_df = pd.DataFrame({
        'SHAP Global': shap_importance,
        'PFI (F1)':    pfi_df.set_index('Feature')['PFI_mean'],
        'RF Native':   rf_importance
    }).fillna(0)

    # Normalizar a [0,1]
    ranking_norm = (ranking_df - ranking_df.min()) / (ranking_df.max() - ranking_df.min())
    ranking_norm['Concordancia'] = ranking_norm.mean(axis=1)
    ranking_norm = ranking_norm.sort_values('Concordancia', ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.heatmap(ranking_norm.drop('Concordancia', axis=1), annot=True, fmt='.3f',
                cmap='RdPu', ax=axes[0], linewidths=0.5,
                cbar_kws={'label': 'Importancia normalizada'})
    axes[0].set_title('Importancia Normalizada por Método XAI', fontweight='bold', fontsize=13)
    axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=0)

    ranking_norm['Concordancia'].plot(kind='barh', ax=axes[1], color=COLORS[0], edgecolor='white')
    axes[1].set_title('Concordancia entre Métodos XAI', fontweight='bold', fontsize=13)
    axes[1].set_xlabel('Importancia promedio normalizada')
    axes[1].invert_yaxis()

    plt.suptitle('🔍 Comparativa XAI: SHAP vs PFI vs RF Native', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()

    top3 = ranking_norm['Concordancia'].head(3).index.tolist()
    print(f"✅ Top 3 features con mayor concordancia: {top3}")
    return ranking_norm

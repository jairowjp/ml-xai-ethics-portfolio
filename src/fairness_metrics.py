"""
fairness_metrics.py
===================
Cálculo de métricas de equidad con Fairlearn y mitigación de sesgo.

MIAR0525 · Semana 4 · Ética, Sesgo y Calidad en ML
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    selection_rate
)
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from typing import Optional

COLORS = ['#8B1A2F', '#C0392B', '#A9CCE3', '#5D6D7E', '#D5D8DC']
SEED = 42


def compute_fairness_metrics(y_true, y_pred, sensitive_features,
                               group_labels: dict = None) -> tuple:
    """
    Calcula métricas de equidad desagregadas por grupo sensible.

    Parameters
    ----------
    y_true : array-like
        Etiquetas reales.
    y_pred : array-like
        Predicciones del modelo.
    sensitive_features : array-like
        Atributo sensible (e.g., sex: 0=Mujer, 1=Hombre).
    group_labels : dict, optional
        Mapeo de valores a nombres de grupos.

    Returns
    -------
    tuple : (mf, resultado, dpd, eod)
        - mf: MetricFrame
        - resultado: pd.DataFrame con métricas por grupo
        - dpd: float (Demographic Parity Difference)
        - eod: float (Equalized Odds Difference)
    """
    if group_labels is None:
        group_labels = {0: 'Mujer', 1: 'Hombre'}

    metricas = {
        'accuracy':       accuracy_score,
        'precision':      lambda y_t, y_p: precision_score(y_t, y_p, zero_division=0),
        'recall':         lambda y_t, y_p: recall_score(y_t, y_p, zero_division=0),
        'f1':             lambda y_t, y_p: f1_score(y_t, y_p, zero_division=0),
        'selection_rate': selection_rate
    }

    mf = MetricFrame(
        metrics=metricas,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features
    )

    resultado = mf.by_group.rename(index=group_labels)
    dpd = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features)
    eod = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_features)

    print("📊 MÉTRICAS DE EQUIDAD POR GRUPO")
    print("=" * 55)
    print(resultado.round(4).to_string())
    print()
    print(f"📌 |DPD| = {abs(dpd):.4f}  |  |EOD| = {abs(eod):.4f}")
    print(f"   DPD > 0.10 indica disparidad significativa.")

    return mf, resultado, dpd, eod


def plot_fairness_metrics(mf: MetricFrame, resultado: pd.DataFrame,
                           dpd: float, eod: float,
                           save_path: Optional[str] = None):
    """Visualiza las métricas de equidad."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Métricas por grupo
    resultado[['accuracy', 'recall', 'f1']].plot(
        kind='bar', ax=axes[0], color=COLORS[:3], edgecolor='white', width=0.7)
    axes[0].set_title('Métricas por Grupo', fontweight='bold')
    axes[0].set_ylim(0, 1.05)
    axes[0].tick_params(rotation=0)
    axes[0].set_ylabel('Valor')
    axes[0].legend(loc='lower right')

    # Selection Rate
    resultado['selection_rate'].plot(
        kind='bar', ax=axes[1], color=[COLORS[1], COLORS[3]],
        edgecolor='white', width=0.45)
    global_sr = mf.overall['selection_rate']
    axes[1].axhline(global_sr, color='gray', linestyle='--',
                   label=f'Global: {global_sr:.3f}')
    axes[1].set_title(f'Selection Rate\n(DPD = {dpd:.4f})', fontweight='bold')
    axes[1].tick_params(rotation=0)
    axes[1].legend()
    for p in axes[1].patches:
        axes[1].annotate(f'{p.get_height():.3f}',
                         (p.get_x() + p.get_width()/2., p.get_height() + 0.003),
                         ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Índices de disparidad
    bars = axes[2].bar(['|DPD|', '|EOD|'], [abs(dpd), abs(eod)],
                       color=[COLORS[0], COLORS[1]], edgecolor='white', width=0.35)
    axes[2].axhline(0.10, color='orange', linestyle='--', alpha=0.8, label='Umbral 0.10')
    axes[2].axhline(0.05, color='green', linestyle='--', alpha=0.8, label='Umbral ideal')
    axes[2].set_title('Índices de Disparidad\n(más bajo = más equitativo)', fontweight='bold')
    axes[2].set_ylabel('|Diferencia|')
    axes[2].legend(fontsize=9)
    for bar, val in zip(bars, [abs(dpd), abs(eod)]):
        axes[2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.003,
                     f'{val:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.suptitle('⚖️ Análisis de Equidad', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()


def train_fair_model(X_train, y_train, sa_train,
                      eps: float = 0.02, max_iter: int = 50,
                      random_state: int = SEED):
    """
    Entrena un modelo con restricción de Paridad Demográfica usando Fairlearn.

    Parameters
    ----------
    X_train : pd.DataFrame
    y_train : pd.Series
    sa_train : pd.Series
        Atributo sensible en train.
    eps : float
        Tolerancia de disparidad (default: 0.02).

    Returns
    -------
    ExponentiatedGradient
        Modelo mitigado entrenado.
    """
    base_clf = DecisionTreeClassifier(max_depth=8, min_samples_leaf=10,
                                       random_state=random_state)
    mitigator = ExponentiatedGradient(
        estimator=base_clf,
        constraints=DemographicParity(),
        eps=eps,
        max_iter=max_iter
    )
    print("⏳ Entrenando modelo con restricción de Paridad Demográfica...")
    mitigator.fit(X_train, y_train, sensitive_features=sa_train)
    print("✅ Modelo mitigado entrenado.")
    return mitigator


def compare_base_vs_mitigated(y_test, y_pred_base, y_pred_fair, sa_test,
                                save_path: Optional[str] = None):
    """
    Compara modelo base vs mitigado en accuracy y equidad.

    Returns
    -------
    pd.DataFrame : Tabla comparativa
    """
    dpd_base = demographic_parity_difference(y_test, y_pred_base, sensitive_features=sa_test)
    dpd_fair = demographic_parity_difference(y_test, y_pred_fair, sensitive_features=sa_test)
    acc_base = accuracy_score(y_test, y_pred_base)
    acc_fair = accuracy_score(y_test, y_pred_fair)
    f1_base  = f1_score(y_test, y_pred_base)
    f1_fair  = f1_score(y_test, y_pred_fair)
    reduccion = (abs(dpd_base) - abs(dpd_fair)) / abs(dpd_base) * 100

    comparacion = pd.DataFrame({
        'Modelo Base (RF)': [f"{acc_base:.4f}", f"{f1_base:.4f}", f"{abs(dpd_base):.4f}"],
        'Modelo Mitigado':  [f"{acc_fair:.4f}", f"{f1_fair:.4f}", f"{abs(dpd_fair):.4f}"]
    }, index=['Accuracy', 'F1 (>50K)', '|DPD|'])

    print("📊 COMPARACIÓN: BASE vs MITIGADO")
    print("=" * 50)
    print(comparacion.to_string())
    print(f"\n✅ Reducción de disparidad: {reduccion:.1f}%")
    print(f"   Costo en accuracy: {(acc_base - acc_fair)*100:.2f}pp")

    if save_path:
        _plot_tradeoff(acc_base, acc_fair, abs(dpd_base), abs(dpd_fair), save_path)

    return comparacion, reduccion


def _plot_tradeoff(acc_base, acc_fair, dpd_base, dpd_fair, save_path=None):
    """Plot interno del trade-off accuracy vs equidad."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    modelos = ['Base\n(RF)', 'Mitigado\n(EG)']
    colors_m = [COLORS[0], COLORS[3]]

    axes[0].bar(modelos, [acc_base, acc_fair], color=colors_m, edgecolor='white', width=0.4)
    axes[0].set_ylim(0.75, 0.92)
    axes[0].set_title('Accuracy Global', fontweight='bold')
    for bar, val in zip(axes[0].patches, [acc_base, acc_fair]):
        axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                     f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    axes[1].bar(modelos, [dpd_base, dpd_fair], color=colors_m, edgecolor='white', width=0.4)
    axes[1].axhline(0.10, color='orange', linestyle='--', alpha=0.8, label='Umbral 0.10')
    axes[1].axhline(0.05, color='green', linestyle='--', alpha=0.8, label='Umbral ideal')
    axes[1].set_title('|Paridad Demográfica|\n(más bajo = más equitativo)', fontweight='bold')
    axes[1].legend(fontsize=9)
    for bar, val in zip(axes[1].patches, [dpd_base, dpd_fair]):
        axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.003,
                     f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.suptitle('⚖️ Trade-off: Accuracy vs Equidad', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()

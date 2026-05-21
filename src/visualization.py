"""
visualization.py
================
Funciones de visualización reutilizables para el Portfolio XAI.
Proyecto: MIAR0525 — Semana 4: Ética, Sesgo y XAI
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ─── Paleta institucional ───────────────────────────────────────────────────
COLORS     = ['#8B1A2F', '#C0392B', '#E74C3C', '#5D6D7E', '#2C3E50']
COLOR_POS  = '#8B1A2F'   # granate UEES — clase positiva
COLOR_NEG  = '#5D6D7E'   # gris azulado — clase negativa
COLOR_WARN = '#E67E22'   # naranja — advertencia


# ─────────────────────────────────────────────────────────────────────────────
# Distribuciones
# ─────────────────────────────────────────────────────────────────────────────

def plot_target_distribution(y: pd.Series, class_names: list = ['≤50K', '>50K'],
                              title: str = "Distribución de la Variable Objetivo",
                              save_path: str = None) -> None:
    """
    Gráfico combinado: barras + torta + boxplot de la variable objetivo.
    """
    conteo = y.value_counts()
    pct    = (conteo / len(y) * 100).round(1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontweight='bold', fontsize=13)

    # Barras
    bars = axes[0].bar(class_names, conteo.values,
                       color=COLORS[:2], edgecolor='white', width=0.5)
    for bar, p in zip(bars, pct.values):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 200, f'{p}%',
                     ha='center', va='bottom', fontweight='bold')
    axes[0].set_title('Conteo por clase')
    axes[0].set_ylabel('Número de instancias')

    # Torta
    axes[1].pie(conteo.values, labels=class_names, colors=COLORS[:2],
                autopct='%1.1f%%', startangle=90,
                wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
    axes[1].set_title(f'Proporción (n={len(y):,})')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_group_bias(df: pd.DataFrame, sensitive_col: str,
                    target_col: str = 'income', positive_class: str = '>50K',
                    title: str = None, save_path: str = None) -> pd.DataFrame:
    """
    Visualiza sesgo por grupos en una variable sensible.

    Returns
    -------
    DataFrame con % de clase positiva por grupo
    """
    cross = df.groupby([sensitive_col, target_col]).size().unstack(fill_value=0)
    pct   = cross.div(cross.sum(axis=1), axis=0) * 100

    if title is None:
        title = f"Distribución de '{positive_class}' por {sensitive_col}"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontweight='bold', fontsize=13)

    pct.plot(kind='bar', ax=axes[0], color=COLORS[:2], edgecolor='white', width=0.7)
    axes[0].set_title('Proporción por grupo (%)')
    axes[0].set_xlabel(sensitive_col)
    axes[0].set_ylabel('Porcentaje')
    axes[0].tick_params(axis='x', rotation=30)
    axes[0].legend([f'≤50K', f'>50K'])

    # Brecha respecto al promedio
    media = pct[positive_class].mean() if positive_class in pct.columns else pct.iloc[:, 1].mean()
    vals  = pct[positive_class] if positive_class in pct.columns else pct.iloc[:, 1]
    gap   = vals - media
    colors_gap = [COLOR_POS if g >= 0 else COLOR_NEG for g in gap]
    gap.plot(kind='bar', ax=axes[1], color=colors_gap, edgecolor='white', width=0.7)
    axes[1].axhline(0, color='black', linewidth=1, linestyle='--')
    axes[1].set_title(f'Brecha respecto al promedio ({media:.1f}%)')
    axes[1].set_xlabel(sensitive_col)
    axes[1].set_ylabel('Puntos porcentuales')
    axes[1].tick_params(axis='x', rotation=30)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    return pct


# ─────────────────────────────────────────────────────────────────────────────
# Fairness
# ─────────────────────────────────────────────────────────────────────────────

def plot_fairness_comparison(resultado_base: pd.DataFrame,
                              resultado_fair: pd.DataFrame,
                              dpd_base: float, dpd_fair: float,
                              acc_base: float, acc_fair: float,
                              save_path: str = None) -> None:
    """
    Comparación visual: modelo base vs modelo mitigado.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Trade-off Equidad–Rendimiento: Base vs Mitigado',
                 fontweight='bold', fontsize=13)

    modelos   = ['Base (RF)', 'Mitigado (EG)']
    acc_vals  = [acc_base, acc_fair]
    dpd_vals  = [abs(dpd_base), abs(dpd_fair)]

    # Accuracy
    bars = axes[0].bar(modelos, acc_vals, color=[COLOR_POS, COLOR_NEG],
                       edgecolor='white', width=0.5)
    for bar, v in zip(bars, acc_vals):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() - 0.02, f'{v:.3f}',
                     ha='center', va='top', color='white', fontweight='bold')
    axes[0].set_ylim(0.75, 0.95)
    axes[0].set_title('Accuracy Global')
    axes[0].set_ylabel('Accuracy')

    # |DPD|
    bars2 = axes[1].bar(modelos, dpd_vals, color=[COLOR_WARN, COLOR_NEG],
                        edgecolor='white', width=0.5)
    for bar, v in zip(bars2, dpd_vals):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.005, f'{v:.3f}',
                     ha='center', va='bottom', fontweight='bold')
    axes[1].axhline(0.10, color='red', linestyle='--', linewidth=1.5,
                    label='Umbral aceptable (0.10)')
    axes[1].set_title('|Paridad Demográfica (DPD)|')
    axes[1].set_ylabel('|DPD|')
    axes[1].legend()

    # Recall por género
    groups = resultado_base.index.tolist()
    x = np.arange(len(groups))
    w = 0.35
    rec_base = resultado_base['recall'].values
    rec_fair = resultado_fair['recall'].values
    axes[2].bar(x - w/2, rec_base, w, label='Base', color=COLOR_POS, edgecolor='white')
    axes[2].bar(x + w/2, rec_fair, w, label='Mitigado', color=COLOR_NEG, edgecolor='white')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(groups)
    axes[2].set_title('Recall por Género: Base vs Mitigado')
    axes[2].set_ylabel('Recall')
    axes[2].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# XAI
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance_comparison(importances: dict,
                                        feature_names: list,
                                        title: str = "Comparativa de Importancia de Variables",
                                        save_path: str = None) -> pd.DataFrame:
    """
    Heatmap normalizado comparando múltiples métodos XAI.

    Parameters
    ----------
    importances  : dict {método: array de importancias}, e.g.
                   {'SHAP': shap_vals, 'PFI': pfi_vals, 'RF Native': rf_imp}
    feature_names: nombres de las variables
    """
    df = pd.DataFrame(importances, index=feature_names)

    # Normalizar 0-1 por columna
    df_norm = df.div(df.max(axis=0), axis=1)

    # Ranking
    ranking = pd.DataFrame({m: df[m].rank(ascending=False).astype(int)
                             for m in df.columns}, index=feature_names)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontweight='bold', fontsize=13)

    sns.heatmap(df_norm, annot=True, fmt='.2f', cmap='RdYlGn',
                ax=axes[0], linewidths=0.5, vmin=0, vmax=1)
    axes[0].set_title('Importancias Normalizadas (0-1)')
    axes[0].tick_params(axis='x', rotation=15)

    sns.heatmap(ranking, annot=True, fmt='d', cmap='RdYlGn_r',
                ax=axes[1], linewidths=0.5)
    axes[1].set_title('Ranking por Método (1=más importante)')
    axes[1].tick_params(axis='x', rotation=15)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    return ranking


def plot_pdp(model, X: pd.DataFrame, feature: str,
             feature_label: str = None, n_grid: int = 50,
             save_path: str = None) -> None:
    """
    Partial Dependence Plot (PDP) para una variable.

    Parameters
    ----------
    model     : modelo con predict_proba
    X         : DataFrame de test
    feature   : nombre de la variable
    n_grid    : puntos en la grilla
    """
    from sklearn.inspection import partial_dependence

    if feature_label is None:
        feature_label = feature

    feature_idx = list(X.columns).index(feature)
    pdp_result  = partial_dependence(model, X, features=[feature_idx],
                                      grid_resolution=n_grid, kind='average')
    grid_vals   = pdp_result['grid_values'][0]
    avg_pred    = pdp_result['average'][0]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(grid_vals, avg_pred, color=COLOR_POS, linewidth=2.5)
    ax.fill_between(grid_vals, avg_pred.min(), avg_pred,
                    alpha=0.15, color=COLOR_POS)
    ax.axhline(avg_pred.mean(), color='gray', linestyle='--',
               linewidth=1, label=f'Media: {avg_pred.mean():.3f}')
    ax.set_xlabel(feature_label, fontsize=12)
    ax.set_ylabel('P(income > 50K) — efecto marginal', fontsize=11)
    ax.set_title(f'Partial Dependence Plot: {feature_label}',
                 fontweight='bold', fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

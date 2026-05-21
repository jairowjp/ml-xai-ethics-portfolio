"""
src/
====
Paquete de utilidades para el Portfolio XAI — MIAR0525 Semana 4.

Módulos disponibles
-------------------
data_loader      : Carga, auditoría y preprocesamiento del Adult Income dataset
model_training   : Entrenamiento y evaluación de modelos supervisados
fairness_metrics : Métricas de equidad (Fairlearn) y mitigación de sesgo
xai_utils        : Explicabilidad con SHAP, LIME y Permutation Feature Importance
visualization    : Funciones de visualización reutilizables
"""

from .data_loader      import load_and_clean_data, audit_quality
from .model_training   import (train_random_forest, train_logistic_regression,
                                train_decision_tree, evaluate_model,
                                cross_validate_model, plot_logistic_coefficients)
from .fairness_metrics import (compute_fairness_metrics, plot_fairness_metrics,
                                train_fair_model, compare_base_vs_mitigated)
from .xai_utils        import (run_shap_analysis, run_lime_analysis,
                                compute_permutation_importance, compare_xai_methods)
from .visualization    import (plot_target_distribution, plot_group_bias,
                                plot_fairness_comparison,
                                plot_feature_importance_comparison, plot_pdp)

__version__ = "1.0.0"
__authors__  = [
    "Daniel Fernando Salgado Santamaría",
    "Jairo Wladimir Jhayya Perlaza",
    "Luis Gabriel Salgado Santamaría",
    "Oscar Paul Naranjo Castro",
]

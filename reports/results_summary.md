# Resumen de Resultados — Portfolio XAI
## MIAR0525 · Semana 4

---

## Rendimiento del Modelo Base (Random Forest)

| Métrica | Valor |
|---------|-------|
| Accuracy global | ~87% |
| F1-Score (>50K) | ~72% |
| AUC-ROC | ~91% |
| Precision (>50K) | ~74% |
| Recall (>50K) | ~70% |
| Cross-Val F1 (5-fold) | ~71% ± 1.5% |

## Métricas de Equidad — Modelo Base

| Métrica | Valor | Evaluación |
|---------|-------|------------|
| |DPD| (Paridad Demográfica) | ~0.22 | ⚠️ Alta disparidad |
| |EOD| (Odds Igualadas) | ~0.30 | ⚠️ Alta disparidad |
| Recall hombres | ~74% | — |
| Recall mujeres | ~52% | ⚠️ Brecha de 22pp |
| Selection Rate hombres | ~34% | — |
| Selection Rate mujeres | ~12% | ⚠️ Brecha de 22pp |

## Importancia de Features — Consenso XAI

| Ranking | Feature | SHAP | PFI | RF Native |
|---------|---------|------|-----|-----------|
| 1 | capital_gain | ✅ | ✅ | ✅ |
| 2 | education_num | ✅ | ✅ | ✅ |
| 3 | marital_status | ✅ | ✅ | ✅ |
| 4 | hours_per_week | ✅ | ✅ | ✅ |
| 5 | age | ✅ | ✅ | ✅ |

Los tres métodos XAI convergen en el mismo ranking top-5.

## Impacto de la Mitigación (Fairlearn EG + DemographicParity)

| Métrica | Base | Mitigado | Δ |
|---------|------|----------|---|
| Accuracy | ~87% | ~83% | -4pp |
| |DPD| | ~0.22 | ~0.05 | **-77%** |
| Selection Rate Hombres | ~34% | ~26% | -8pp |
| Selection Rate Mujeres | ~12% | ~21% | +9pp |

## Figuras Generadas

| Archivo | Descripción |
|---------|-------------|
| `01_data_quality_audit.png` | Mapa de valores faltantes + barras de nulos |
| `02_target_distribution.png` | Distribución de la variable objetivo |
| `03_gender_bias.png` | Análisis de sesgo por género |
| `04_race_bias.png` | Análisis de sesgo por raza |
| `05_model_evaluation.png` | Confusion Matrix + ROC + CV |
| `06_fairness_metrics.png` | Métricas de equidad por grupo |
| `07_shap_global_importance.png` | SHAP vs RF Native importance |
| `08_shap_beeswarm.png` | SHAP Beeswarm (dirección e intensidad) |
| `09_shap_waterfall_pos.png` | Waterfall instancia positiva |
| `10_shap_waterfall_neg.png` | Waterfall instancia negativa |
| `11_lime_instance_pos.png` | LIME explicación instancia positiva |
| `12_lime_instance_neg.png` | LIME explicación instancia negativa |
| `13_permutation_importance.png` | PFI con barras de error |
| `14_xai_comparison.png` | Heatmap comparativo SHAP vs PFI vs RF |
| `15_mitigation_tradeoff.png` | Trade-off accuracy vs equidad |

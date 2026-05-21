# Model Card — RF-Income-Classifier v1.0

*Siguiendo el estándar de Model Cards (Mitchell et al., 2019)*

---

## Información del Modelo

| Campo | Valor |
|-------|-------|
| **Nombre** | RF-Income-Classifier v1.0 |
| **Tipo de modelo** | Random Forest Classifier |
| **Framework** | scikit-learn 1.4 |
| **Versión** | 1.0.0 |
| **Fecha de entrenamiento** | Mayo 2026 |
| **Desarrolladores** | Daniel Salgado, Jairo Jhayya, Luis Salgado, Oscar Naranjo |
| **Institución** | Universidad de los Hemisferios — MIAR0525 |

---

## Uso Previsto

### Usos primarios
- **Investigación académica** sobre ética y sesgo en ML
- **Demostración de técnicas XAI** (SHAP, LIME, PFI)
- **Educación** sobre fairness en aprendizaje automático

### Usos fuera de alcance
- ❌ Decisiones crediticias reales sin auditoría de equidad previa
- ❌ Evaluación de candidatos en procesos de selección laboral
- ❌ Cualquier aplicación en América Latina sin revalidación completa
- ❌ Decisiones que afecten a poblaciones vulnerables sin supervisión humana

---

## Datos de Entrenamiento

| Aspecto | Descripción |
|---------|-------------|
| **Dataset** | Adult Income (UCI ML Repository) |
| **Fuente** | Censo de EE.UU. — 1994 |
| **Tamaño** | 48,842 instancias (tras limpieza: ~48,000) |
| **Features usadas** | 9 de 14 (excluye atributos sensibles directos) |
| **Variable objetivo** | `income` — binaria: ≤50K vs >50K |
| **Balance de clases** | ~76% (≤50K) vs ~24% (>50K) |

### Limitaciones conocidas del dataset
- **Temporal:** Datos de 1994 — no refleja la realidad laboral actual
- **Geográfico:** Exclusivamente EE.UU. — no generalizable a otros contextos
- **Representatividad:** Blancos: ~85%, hombres: ~67% (subrepresentación de minorías)
- **Definición de income:** USD 50K en 1994 ≠ USD 50K en 2026 (inflación)

---

## Evaluación del Modelo

### Rendimiento global

| Métrica | Valor |
|---------|-------|
| Accuracy | ~87% |
| F1-Score (>50K) | ~72% |
| AUC-ROC | ~91% |
| Precision (>50K) | ~74% |
| Recall (>50K) | ~70% |

*Evaluado en 20% del dataset separado antes del entrenamiento (n≈9,600)*

### Rendimiento por subgrupo

| Grupo | Accuracy | Recall (>50K) | Selection Rate |
|-------|----------|---------------|----------------|
| Hombres | ~87% | ~74% | ~34% |
| Mujeres | ~86% | ~52% | ~12% |

### Métricas de equidad

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| Diferencia de Paridad Demográfica (DPD) | ~0.22 | ⚠️ Alta disparidad (> umbral 0.10) |
| Diferencia de Odds Igualadas (EOD) | ~0.30 | ⚠️ Alta disparidad |
| Brecha de Recall por género | ~22 pp | ⚠️ Significativa |

---

## Consideraciones Éticas

### Sesgos identificados

1. **Sesgo de género:** El modelo predice >50K con 22pp mayor frecuencia para hombres
2. **Proxy discrimination:** `capital_gain` actúa como proxy de clase social y potencialmente de raza/género
3. **Sesgo histórico:** Los patrones aprendidos reflejan desigualdades de 1994

### Riesgos de uso

- **Alto riesgo:** Uso en decisiones crediticias, de selección de personal o beneficios sociales sin auditoría
- **Riesgo medio:** Uso como referencia educativa sin contextualizar las limitaciones
- **Bajo riesgo:** Investigación académica con pleno conocimiento de las limitaciones

### Mitigación disponible

Se ha entrenado una versión mitigada con Fairlearn (Exponentiated Gradient + DemographicParity):
- Reduce DPD de ~0.22 a ~0.05 (reducción del ~77%)
- Costo: ~4pp de accuracy global
- **Recomendación:** Usar el modelo mitigado si alguna aplicación real se considera

---

## Explicabilidad

| Técnica | Tipo | Hallazgo principal |
|---------|------|-------------------|
| SHAP | Global + Local | `capital_gain` es la feature más importante globalmente |
| LIME | Local | Confirma la dominancia de `capital_gain` y `education_num` |
| Permutation FI | Global | Valida el ranking de SHAP con garantías más robustas |

Los tres métodos convergen en identificar `capital_gain`, `education_num` y `marital_status` como el núcleo decisional del modelo.

---

## Recomendaciones para Despliegue Responsable

Si este modelo (o uno similar) se considerara para uso real:

1. ✅ Reentrenar con datos actualizados y representativos del contexto destino
2. ✅ Realizar auditoría de equidad extendida (más atributos sensibles)
3. ✅ Aplicar mitigación de sesgo antes del despliegue
4. ✅ Implementar monitoreo continuo de métricas de equidad
5. ✅ Proveer explicaciones individuales (LIME/SHAP) con cada decisión
6. ✅ Establecer proceso de apelación humana para decisiones adversas
7. ✅ Obtener revisión legal de cumplimiento con regulaciones locales
8. ✅ Publicar auditorías de equidad para transparencia

---

## Población para la cual el modelo NO se recomienda

- Personas en países fuera de EE.UU. (sin revalidación)
- Decisiones que afectan a grupos históricamente marginalizados sin auditoría adicional
- Aplicaciones en tiempo real sin supervisión humana
- Cualquier contexto donde DPD > 0.10 sea inaceptable legalmente

---

## Información de Contacto

**Repositorio:** [GitHub — ml-xai-ethics-portfolio](https://github.com/TU_USUARIO/ml-xai-ethics-portfolio)  
**Institución:** Universidad de los Hemisferios  
**Curso:** MIAR0525 — Aprendizaje Automático  

---

## Referencias

- Mitchell, M., Wu, S., Zaldivar, A., et al. (2019). Model Cards for Model Reporting. *FAccT 2019*.
- Gebru, T., Morgenstern, J., Vecchione, B., et al. (2021). Datasheets for Datasets. *Communications of the ACM*.
- Barocas, S., Hardt, M., & Narayanan, A. (2023). *Fairness and Machine Learning*. MIT Press.

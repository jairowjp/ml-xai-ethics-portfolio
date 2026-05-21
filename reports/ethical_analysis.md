# Análisis Ético Completo — Portfolio XAI
## MIAR0525 · Semana 4 · Ética, Sesgo y Calidad en el Aprendizaje Automático

**Equipo:** Daniel Salgado · Jairo Jhayya · Luis Salgado · Oscar Naranjo  
**Fecha:** Mayo 2026  
**Modelo analizado:** Random Forest Classifier — Adult Income (UCI)

---

## 1. Transparencia del Modelo

### 1.1 ¿Qué revelan las técnicas XAI?

Las tres técnicas XAI aplicadas (SHAP, LIME y Permutation Feature Importance) convergen en identificar un conjunto reducido de variables dominantes:

**Top features por consenso XAI:**
1. `capital_gain` — Mayor impacto global (SHAP + PFI + RF Native)
2. `education_num` — Segunda variable más influyente de forma consistente
3. `marital_status` — Alta concordancia entre los tres métodos
4. `hours_per_week` — Cuarta posición con variabilidad moderada entre métodos
5. `age` — Impacto significativo pero menor que las anteriores

**Interpretación a nivel global (SHAP Beeswarm):**
- `capital_gain` alto empuja fuertemente la predicción hacia `>50K`
- `education_num` alto tiene relación positiva con ingresos altos
- `marital_status` muestra efectos bidireccionales según el valor específico
- `capital_loss` presenta un comportamiento paradójico: valores moderados de pérdida de capital correlacionan con ingresos altos (indicando participación en mercados de inversión)

**Interpretación a nivel local (LIME + SHAP Waterfall):**
Para instancias individuales, los factores dominantes varían, pero `capital_gain` y `education_num` mantienen consistencia. Esto es crucial: significa que si un individuo tiene capital_gain = 0 (la mayoría), la decisión recae casi completamente en `education_num`, `marital_status` y `occupation`.

### 1.2 Implicaciones de la falta de explicabilidad

Si este modelo se desplegara sin herramientas XAI:
- Una persona rechazada en una solicitud de crédito no podría conocer el motivo
- Los auditores no podrían verificar si el modelo discrimina
- Sería imposible cumplir con regulaciones como el GDPR (Art. 22) o la Ley de IA de la UE
- Los sesgos detectados (DPD=0.22) pasarían desapercibidos y causarían daño sistemático

---

## 2. Análisis de Sesgos Detectados

### 2.1 Sesgo en los Datos (Pre-modelo)

**Sesgo de muestreo por género:**
- Hombres: 67% del dataset | Mujeres: 33%
- Hombres con >50K: ~30% | Mujeres con >50K: ~11%
- Brecha en el dataset: ~19 puntos porcentuales

**Sesgo de muestreo por raza:**
- Blancos representan ~85% del dataset
- La brecha máxima entre razas en % de >50K supera los 20 puntos porcentuales

Estos sesgos en los datos son **previos al modelo**. El algoritmo los aprenderá y potencialmente los amplificará.

### 2.2 Sesgo del Modelo (Post-entrenamiento)

**Paridad Demográfica (DPD = 0.2198):**
El modelo predice ingresos altos con 22 puntos porcentuales de mayor frecuencia para hombres que para mujeres. Esto supera ampliamente el umbral de equidad aceptable (0.10).

**Igualdad de Oportunidades (EOD ≈ 0.30):**
La tasa de verdaderos positivos (recall para >50K) es significativamente más alta para hombres que para mujeres. El modelo es menos "generoso" al identificar mujeres que realmente ganan >50K.

**Recall por género:**
- Hombres: ~74% de los casos >50K identificados correctamente
- Mujeres: ~52% de los casos >50K identificados correctamente

Este hallazgo es especialmente problemático: incluso cuando una mujer SÍ gana más de 50K, el modelo tiene ~48% de probabilidad de clasificarla incorrectamente como de ingresos bajos.

### 2.3 Variables Proxy y Discriminación Indirecta

Un hallazgo crítico del análisis SHAP es el rol de `capital_gain` como **variable proxy**:

- `capital_gain` refleja ganancias por inversiones en capital
- El acceso a inversiones históricamente ha sido desigual por clase social, raza y género
- Al ser la variable más importante del modelo, actúa como proxy indirecto de atributos protegidos
- **Eliminar `race` y `sex` del dataset no elimina el sesgo** mientras existan estas correlaciones

Lo mismo aplica a `marital_status`: históricamente, el estado civil correlaciona con género (más mujeres históricamente identificadas como "casadas" y dependientes económicamente).

---

## 3. Análisis Comparativo de Técnicas XAI

### 3.1 SHAP vs LIME: ¿Coinciden?

Para las instancias analizadas:
- **Coincidencia:** Ambos identifican `capital_gain` y `education_num` como factores dominantes
- **Divergencia:** LIME puede atribuir mayor importancia a condiciones de umbral locales que SHAP no captura globalmente
- **Recomendación:** Usar SHAP para análisis global y auditorías regulares; LIME para comunicar decisiones individuales a personas no técnicas

### 3.2 Fiabilidad de cada método

| Criterio | SHAP | LIME | PFI |
|----------|------|------|-----|
| Consistencia entre ejecuciones | Alta | Media | Alta |
| Coste computacional | Medio-alto | Bajo | Medio |
| Garantías teóricas | Sí (Shapley) | No | No |
| Adecuado para auditorías | Sí | Complementario | Sí |
| Adecuado para comunicación | Medio | Alto | Medio |

---

## 4. Reflexiones Éticas Fundamentadas

### Pregunta 1: ¿Es válido usar el Adult Income dataset en 2026 en América Latina?

**No, sin revalidación exhaustiva.** El dataset proviene del censo de EE.UU. de 1994, lo que introduce múltiples problemas:

- **Temporalidad:** Las relaciones entre variables han cambiado en 30 años (mayor participación femenina en el mercado laboral, cambios en estructuras familiares, etc.)
- **Contexto geográfico:** Las estructuras salariales, de acceso a educación y distribución de riqueza de EE.UU. 1994 no son transferibles a América Latina 2026
- **Tipos de sesgo introducidos:** Sesgo histórico (perpetúa desigualdades de los 90s), sesgo geográfico (no representa realidades latinoamericanas), sesgo de supervivencia (solo captura personas que participaron en el censo)

**Si se usara en América Latina:** Se tomarían decisiones basadas en patrones de un país diferente, en una época diferente, con estructuras socioeconómicas fundamentalmente distintas. El daño potencial es significativo.

### Pregunta 2: ¿En qué aplicaciones aceptarías el trade-off accuracy-equidad?

**Contextos donde la equidad tiene prioridad absoluta:**

- **Crédito bancario:** Las regulaciones (GDPR, Fair Lending Act, etc.) exigen demostrar no-discriminación. Un modelo que discrimina por género viola la ley, independientemente de su accuracy. El trade-off de -4% en accuracy es completamente aceptable si reduce el DPD de 0.22 a 0.05.

- **Selección de personal:** La discriminación algorítmica en contratación tiene consecuencias legales y reputacionales severas. La equidad debe primar. Amazon pagó el precio de no aplicar este principio en 2018.

- **Diagnóstico médico:** Aquí el balance es más complejo. La equidad entre grupos es crucial (el modelo debe diagnosticar igualmente bien a todos los grupos demográficos), pero una caída en accuracy puede costar vidas. En este caso, el objetivo es maximizar AMBAS métricas, no sacrificar una por otra.

**Principios éticos que guían esta decisión:**
- **Justicia:** La distribución inequitativa de beneficios y riesgos es inaceptable en sistemas que afectan el acceso a recursos
- **Beneficencia:** Un sistema que causa daño diferencial a grupos vulnerables no maximiza el beneficio colectivo
- **Autonomía:** Las personas afectadas por decisiones algorítmicas deben tener acceso a recursos de impugnación

### Pregunta 3: ¿Cómo manejar `capital_gain` como variable proxy?

Eliminar `race` y `sex` NO es suficiente. Se requiere un enfoque más profundo:

1. **Análisis de correlación:** Medir la correlación entre `capital_gain` y atributos protegidos para cuantificar el problema
2. **Fairness-aware feature selection:** Usar técnicas que penalicen features con alta correlación con atributos protegidos
3. **Counterfactual fairness:** Preguntar: "¿Cambiaría la predicción si este individuo fuera de diferente género/raza con las mismas capacidades objetivas?"
4. **Causal reasoning:** Distinguir entre correlación espuria (hombre→alto capital_gain→>50K) y capacidad real
5. **Documentar la decisión:** Si se mantiene `capital_gain`, documentar explícitamente el riesgo de proxy discrimination

### Pregunta 4: Caso Real de Daño por Sesgo Algorítmico

**COMPAS (Correctional Offender Management Profiling for Alternative Sanctions) — ProPublica, 2016**

**Tipo de sesgo:** Sesgo racial en predicción de reincidencia criminal

**Descripción:** COMPAS es un sistema de ML utilizado en tribunales de EE.UU. para predecir la probabilidad de reincidencia criminal. ProPublica investigó 7,000 casos en Florida y encontró que el algoritmo:
- Clasificaba a acusados negros como de alto riesgo de reincidencia el doble de veces que a acusados blancos (cuando no reincidieron)
- Clasificaba a acusados blancos como de bajo riesgo incluso cuando sí reincidieron

**Cómo se detectó:** Análisis estadístico independiente de las predicciones vs. resultados reales, desagregado por raza. Requirió acceso a los datos de predicción (whistleblower) y metodología de auditoría externa.

**Lo que se hizo / debería haberse hecho:**
- *Lo que se hizo:* La empresa (Northpointe) argumentó que el modelo era igualmente preciso para ambos grupos (lo cual era verdad en accuracy global, pero no en tasas de error diferenciadas)
- *Lo que debería haberse hecho:* Auditorías de equidad antes del despliegue; uso de métricas como equalized odds; documentación transparente de limitaciones; supervisión humana en todas las decisiones (el sistema no debería haber reemplazado el juicio judicial sino informarlo)

**Lección:** La accuracy global no es suficiente como métrica de evaluación cuando el modelo afecta a grupos protegidos. La transparencia y las auditorías independientes son fundamentales.

---

## 5. Propuesta de Mejoras al Modelo

### Inmediatas (técnicas)
1. Reentrenamiento con datos más recientes y representativos
2. Auditoría de equidad extendida: incluir `race`, `age`, `native_country` como atributos sensibles
3. Implementación de mitigación (EG con DemographicParity) en producción
4. Monitoreo continuo de DPD y EOD (alertas si superan 0.10)

### Mediano plazo (proceso)
5. Documentación de Model Card y Datasheet completos
6. Implementación de proceso de apelación humana para decisiones adversas
7. Revisión legal de cumplimiento con regulaciones locales
8. Pruebas de robustez a distributional shift

### Largo plazo (sistémica)
9. Recolección de datos propios con diseño de muestreo equitativo
10. Participación de comunidades afectadas en el diseño del sistema
11. Revisión ética independiente periódica
12. Publicación de auditorías de equidad para transparencia pública

---

## 6. Conclusión

El análisis de este portfolio demuestra que construir un sistema de ML responsable requiere mucho más que optimizar métricas de accuracy. Las tres dimensiones son inseparables:

**Calidad de datos → Detección de sesgo → Explicabilidad → Mitigación → Documentación ética**

El modelo Random Forest analizado es técnicamente competente (AUC=0.91) pero éticamente problemático sin mitigación (DPD=0.22). La aplicación de Exponentiated Gradient reduce el sesgo un 77% con un costo de solo ~4% en accuracy, demostrando que la equidad y el rendimiento no son objetivos irreconciliables.

La lección más importante: **la responsabilidad ética en ML no es opcional ni cosmética — es la condición de posibilidad de que estos sistemas generen valor real sin causar daño a los grupos más vulnerables.**

---

*Preparado para MIAR0525 — Semana 4 | Mayo 2026*

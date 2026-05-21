# Datasheet for Datasets — Adult Income (UCI)

*Siguiendo el estándar de Datasheets for Datasets (Gebru et al., 2021)*

---

## Motivación

**¿Por qué se creó este dataset?**  
El dataset Adult Income fue extraído del censo de EE.UU. de 1994 por Ronny Kohavi y Barry Becker, con el objetivo de crear un benchmark para tareas de predicción de ingresos. Fue publicado en el UCI ML Repository para facilitar la investigación en clasificación.

**¿Quién lo creó?**  
Ronny Kohavi (Silicon Graphics) y Barry Becker, 1996.

---

## Composición

| Aspecto | Detalle |
|---------|---------|
| Instancias | 48,842 (train: 32,561 + test: 16,281) |
| Features | 14 (6 continuas + 8 categóricas) |
| Variable objetivo | `income` — binaria: ≤50K vs >50K (USD anuales) |
| Valores faltantes | Sí — en `workclass`, `occupation`, `native_country` (~7%) |
| Duplicados | Sí — ~24 registros |

### Features

| Feature | Tipo | Descripción | Atributo sensible? |
|---------|------|-------------|---------------------|
| age | Continua | Edad en años | Sí (edad puede ser protegida) |
| workclass | Categórica | Tipo de empleador | No |
| fnlwgt | Continua | Peso de muestra del censo | No |
| education | Categórica | Nivel educativo | No |
| education_num | Continua | Años de educación | No |
| marital_status | Categórica | Estado civil | Correlacionada con género |
| occupation | Categórica | Ocupación laboral | Correlacionada con género/raza |
| relationship | Categórica | Rol familiar | Correlacionada con género |
| **race** | Categórica | Raza | **SÍ — atributo protegido** |
| **sex** | Categórica | Género | **SÍ — atributo protegido** |
| capital_gain | Continua | Ganancias de capital | Proxy de clase social |
| capital_loss | Continua | Pérdidas de capital | Proxy de clase social |
| hours_per_week | Continua | Horas trabajadas por semana | Correlacionada con género |
| native_country | Categórica | País de origen | Proxy de etnia |
| **income** | Categórica | **Variable objetivo** | — |

---

## Proceso de Recolección

- **Fuente:** Base de datos del censo de EE.UU. 1994
- **Método:** Extracción de registros del Census Bureau
- **Período:** Datos de 1994 únicamente
- **Geografía:** Exclusivamente Estados Unidos
- **Selección:** Solo personas de 16+ años con salario > 100 USD/semana

---

## Preprocesamiento (aplicado en este proyecto)

1. **Manejo de valores faltantes:** Imputación con moda para variables categóricas (`workclass`, `occupation`, `native_country`)
2. **Eliminación de duplicados:** 24 registros removidos
3. **Codificación:** Label Encoding para variables categóricas
4. **Separación de atributos sensibles:** `sex` y `race` preservados para análisis de equidad, no incluidos como features de entrenamiento

---

## Sesgos Conocidos

| Tipo de sesgo | Descripción |
|--------------|-------------|
| Sesgo de muestreo | Hombres sobrerrepresentados (67%), blancos sobrerrepresentados (85%) |
| Sesgo de confirmación histórico | Los datos reflejan las desigualdades de EE.UU. en 1994 |
| Sesgo de temporalidad | Datos de hace 30 años no representan la realidad laboral actual |
| Sesgo geográfico | Solo EE.UU. — no generalizable a otros países |
| Sesgo de umbral | USD 50K en 1994 ≈ USD 100K en 2026 (ajustado por inflación) |

---

## Usos Recomendados y No Recomendados

### ✅ Usos recomendados
- Investigación académica sobre sesgo y equidad en ML
- Benchmarking de algoritmos de clasificación
- Demostración de técnicas XAI y fairness
- Educación sobre problemas de datos históricos

### ❌ Usos no recomendados
- Decisiones de crédito o préstamos reales
- Selección de personal o evaluación de candidatos
- Determinar elegibilidad para beneficios sociales
- Cualquier aplicación fuera de EE.UU.
- Cualquier aplicación en 2026 sin revalidación temporal

---

## Distribución

- **Licencia:** Public Domain (UCI ML Repository)
- **URL:** https://archive.ics.uci.edu/ml/datasets/adult
- **DOI:** Kohavi, R. (1996). Scaling Up the Accuracy of Naive-Bayes Classifiers: A Decision-Tree Hybrid. *KDD-96*.

---

## Mantenimiento

- **Mantenido por:** UCI ML Repository
- **Última actualización conocida:** Sin actualizaciones desde 1996
- **Recomendación:** Buscar censos más recientes del Census Bureau para aplicaciones actuales

---

*Documento preparado para MIAR0525 — Semana 4 | Mayo 2026*

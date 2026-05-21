"""
data_loader.py
==============
Módulo de carga y preprocesamiento del Adult Income dataset.

MIAR0525 · Semana 4 · Ética, Sesgo y Calidad en ML
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# URL del dataset
ADULT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

COLUMNAS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'sex',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
]

FEATURES = [
    'age', 'education_num', 'hours_per_week',
    'capital_gain', 'capital_loss',
    'workclass', 'marital_status', 'occupation', 'relationship'
]

CATS = ['workclass', 'marital_status', 'occupation', 'relationship']

SEED = 42


def load_raw_data(url: str = ADULT_URL) -> pd.DataFrame:
    """
    Carga el dataset Adult Income desde UCI.

    Returns
    -------
    pd.DataFrame
        Dataset crudo con valores nulos marcados.
    """
    df = pd.read_csv(url, header=None, names=COLUMNAS,
                     na_values=' ?', skipinitialspace=True)
    print(f"✅ Dataset cargado: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    return df


def audit_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera un reporte de auditoría de calidad del dataset.

    Returns
    -------
    pd.DataFrame
        DataFrame con conteo y porcentaje de nulos por columna.
    """
    nulos = df.isnull().sum()
    nulos_pct = (nulos / len(df) * 100).round(2)
    auditoria = pd.DataFrame({'Nulos': nulos, '% del total': nulos_pct})
    auditoria = auditoria[auditoria['Nulos'] > 0]

    print(f"📌 Registros completos: {df.dropna().shape[0]:,} de {df.shape[0]:,}")
    print(f"📌 Registros duplicados: {df.duplicated().sum()}")
    print(f"📌 Columnas con nulos:\n{auditoria.to_string()}")
    return auditoria


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia el dataset: imputa nulos con moda y elimina duplicados.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset crudo.

    Returns
    -------
    pd.DataFrame
        Dataset limpio y listo para modelado.
    """
    df_clean = df.copy()

    # Imputar variables categóricas con moda
    for col in ['workclass', 'occupation', 'native_country']:
        moda = df_clean[col].mode()[0]
        n_na = df_clean[col].isnull().sum()
        df_clean[col].fillna(moda, inplace=True)
        print(f"  ✔ '{col}': {n_na} nulos imputados con moda='{moda}'")

    # Eliminar duplicados
    n_antes = len(df_clean)
    df_clean.drop_duplicates(inplace=True)
    df_clean.reset_index(drop=True, inplace=True)
    print(f"  ✔ Duplicados eliminados: {n_antes - len(df_clean)}")
    print(f"  ✔ Dataset final: {len(df_clean):,} registros")

    return df_clean


def preprocess_for_modeling(df: pd.DataFrame,
                             test_size: float = 0.20,
                             random_state: int = SEED):
    """
    Preprocesa el dataset para entrenamiento: codifica variables,
    separa atributo sensible y realiza train/test split.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset limpio.
    test_size : float
        Fracción del dataset para test (default: 0.20).
    random_state : int
        Semilla aleatoria.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test, sa_train, sa_test, FEATURES)
    """
    df_model = df.copy()

    # Atributo sensible (guardamos antes de codificar)
    sensitive_attr = df_model['sex'].map({'Male': 1, 'Female': 0})

    # Variable objetivo binaria
    df_model['target'] = (df_model['income'] == '>50K').astype(int)

    # Codificación de categorías
    le = LabelEncoder()
    for col in CATS:
        df_model[col] = le.fit_transform(df_model[col].astype(str))

    X = df_model[FEATURES]
    y = df_model['target']

    X_train, X_test, y_train, y_test, sa_train, sa_test = train_test_split(
        X, y, sensitive_attr,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"✅ Split completado:")
    print(f"   Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")
    print(f"   Positivos en train: {y_train.mean()*100:.1f}%")

    return X_train, X_test, y_train, y_test, sa_train, sa_test


def load_and_clean_data(url: str = ADULT_URL, test_size: float = 0.20):
    """
    Pipeline completo: carga, limpieza y preprocesamiento.

    Returns
    -------
    tuple
        (df_clean, X_train, X_test, y_train, y_test, sa_train, sa_test)
    """
    print("=" * 50)
    print("  PIPELINE DE CARGA Y PREPROCESAMIENTO")
    print("=" * 50)

    df_raw   = load_raw_data(url)
    _        = audit_quality(df_raw)
    df_clean = clean_data(df_raw)
    splits   = preprocess_for_modeling(df_clean, test_size=test_size)

    print("\n✅ Pipeline completado exitosamente.")
    return (df_clean,) + splits

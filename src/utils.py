"""
Utilidades para el Challenge de Clasificación de Literatura Médica
"""

import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def preprocess_medical_text(text):
    """
    Preprocesa texto médico manteniendo terminología especializada
    
    Args:
        text (str): Texto a preprocesar
    
    Returns:
        str: Texto preprocesado
    """
    if pd.isna(text):
        return ""
    
    # Convertir a minúsculas
    text = text.lower()
    
    # Remover puntuación pero mantener guiones en términos médicos
    text = re.sub(r'[^\w\s\-]', ' ', text)
    
    # Normalizar espacios
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remover números standalone pero mantener códigos médicos
    text = re.sub(r'\b\d+\b', '', text)
    
    return text

def remove_common_stopwords(text):
    """
    Remueve stopwords comunes pero conserva términos médicos importantes
    
    Args:
        text (str): Texto a filtrar
    
    Returns:
        str: Texto sin stopwords
    """
    # Stopwords médicas específicas que podemos remover
    medical_stopwords = {
        'patient', 'patients', 'study', 'studies', 'method', 'methods',
        'result', 'results', 'conclusion', 'background', 'objective',
        'purpose', 'design', 'participants', 'intervention'
    }
    
    # Combinar con stopwords de sklearn
    all_stopwords = ENGLISH_STOP_WORDS.union(medical_stopwords)
    
    words = text.split()
    filtered_words = [word for word in words if word not in all_stopwords and len(word) > 2]
    
    return ' '.join(filtered_words)

def load_and_preprocess_data(file_path):
    """
    Carga y preprocesa el dataset médico
    
    Args:
        file_path (str): Ruta al archivo CSV
    
    Returns:
        tuple: (DataFrame, etiquetas_multilabel)
    """
    # Cargar datos
    df = pd.read_csv(file_path, sep=';')
    
    # Preprocesar texto
    df['title_clean'] = df['title'].apply(preprocess_medical_text)
    df['abstract_clean'] = df['abstract'].apply(preprocess_medical_text)
    df['combined_clean'] = df['title_clean'] + ' ' + df['abstract_clean']
    df['combined_filtered'] = df['combined_clean'].apply(remove_common_stopwords)
    
    # Preparar etiquetas multi-label
    df['groups_list'] = df['group'].str.split('|')
    
    return df

def get_dataset_statistics(df):
    """
    Genera estadísticas descriptivas del dataset
    
    Args:
        df (DataFrame): Dataset preprocesado
    
    Returns:
        dict: Estadísticas del dataset
    """
    from collections import Counter
    
    # Estadísticas básicas
    stats = {
        'total_records': len(df),
        'total_columns': df.shape[1],
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
    }
    
    # Análisis de categorías
    all_categories = []
    for groups in df['groups_list']:
        all_categories.extend(groups)
    
    category_counts = Counter(all_categories)
    stats['categories'] = dict(category_counts)
    stats['unique_categories'] = len(category_counts)
    
    # Análisis multi-label
    df['num_categories'] = df['groups_list'].apply(len)
    stats['multi_label_distribution'] = df['num_categories'].value_counts().to_dict()
    
    # Estadísticas de texto
    stats['text_stats'] = {
        'avg_title_length': df['title'].str.len().mean(),
        'avg_abstract_length': df['abstract'].str.len().mean(),
        'avg_title_words': df['title'].str.split().str.len().mean(),
        'avg_abstract_words': df['abstract'].str.split().str.len().mean()
    }
    
    return stats

def print_model_summary(model, vectorizer, mlb, metrics):
    """
    Imprime un resumen del modelo entrenado
    
    Args:
        model: Modelo entrenado
        vectorizer: Vectorizador TF-IDF
        mlb: MultiLabelBinarizer
        metrics (dict): Métricas de evaluación
    """
    print("🎯 RESUMEN DEL MODELO")
    print("="*40)
    print(f"Tipo de modelo: {type(model).__name__}")
    print(f"Características TF-IDF: {vectorizer.get_feature_names_out().shape[0]:,}")
    print(f"Categorías soportadas: {', '.join(mlb.classes_)}")
    print(f"F1-Score Macro: {metrics['f1_macro']:.3f}")
    print(f"F1-Score Micro: {metrics['f1_micro']:.3f}")
    print(f"Hamming Loss: {metrics['hamming_loss']:.4f}")
    print("✅ Modelo listo para uso en producción!")

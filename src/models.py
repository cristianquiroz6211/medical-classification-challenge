"""
Modelos para la clasificación de literatura médica
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    hamming_loss, accuracy_score
)
import joblib
import time

from .utils import preprocess_medical_text, remove_common_stopwords


class MedicalTextClassifier:
    """
    Clasificador de texto médico multi-label usando TF-IDF + SVM
    """
    
    def __init__(self, 
                 max_features=10000,
                 min_df=2,
                 max_df=0.95,
                 ngram_range=(1, 2),
                 C=1.0,
                 kernel='linear',
                 random_state=42):
        """
        Inicializa el clasificador médico
        
        Args:
            max_features (int): Número máximo de características TF-IDF
            min_df (int): Frecuencia mínima de documento
            max_df (float): Frecuencia máxima de documento
            ngram_range (tuple): Rango de n-gramas
            C (float): Parámetro de regularización SVM
            kernel (str): Kernel de SVM
            random_state (int): Semilla aleatoria
        """
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.C = C
        self.kernel = kernel
        self.random_state = random_state
        
        # Inicializar componentes
        self.vectorizer = None
        self.model = None
        self.mlb = None
        self.is_trained = False
        
    def _preprocess_text(self, title, abstract):
        """
        Preprocesa título y abstract
        
        Args:
            title (str): Título del artículo
            abstract (str): Abstract del artículo
        
        Returns:
            str: Texto preprocesado y filtrado
        """
        title_clean = preprocess_medical_text(title)
        abstract_clean = preprocess_medical_text(abstract)
        combined_text = title_clean + ' ' + abstract_clean
        filtered_text = remove_common_stopwords(combined_text)
        return filtered_text
        
    def fit(self, df):
        """
        Entrena el modelo con el dataset
        
        Args:
            df (DataFrame): Dataset con columnas 'title', 'abstract', 'groups_list'
        
        Returns:
            dict: Métricas de entrenamiento
        """
        print("🚀 Iniciando entrenamiento del modelo...")
        start_time = time.time()
        
        # Preparar datos
        print("📝 Preparando datos...")
        X = []
        for _, row in df.iterrows():
            processed_text = self._preprocess_text(row['title'], row['abstract'])
            X.append(processed_text)
        
        # Preparar etiquetas multi-label
        self.mlb = MultiLabelBinarizer()
        y = self.mlb.fit_transform(df['groups_list'])
        
        # División de datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        print(f"📊 Datos divididos - Entrenamiento: {len(X_train)}, Prueba: {len(X_test)}")
        
        # Vectorización TF-IDF
        print("🔤 Vectorizando texto...")
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=self.ngram_range,
            stop_words='english',
            lowercase=True,
            strip_accents='ascii'
        )
        
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Entrenamiento del modelo
        print("🤖 Entrenando modelo SVM...")
        self.model = MultiOutputClassifier(
            SVC(
                kernel=self.kernel,
                C=self.C,
                random_state=self.random_state
            ),
            n_jobs=1  # Evitar problemas de paralelismo en Windows
        )
        
        training_start = time.time()
        self.model.fit(X_train_tfidf, y_train)
        training_time = time.time() - training_start
        
        # Evaluación
        print("📈 Evaluando modelo...")
        y_pred = self.model.predict(X_test_tfidf)
        
        # Calcular métricas
        metrics = {
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'f1_micro': f1_score(y_test, y_pred, average='micro'),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
            'precision_micro': precision_score(y_test, y_pred, average='micro'),
            'recall_micro': recall_score(y_test, y_pred, average='micro'),
            'hamming_loss': hamming_loss(y_test, y_pred),
            'training_time': training_time,
            'total_time': time.time() - start_time
        }
        
        # Métricas por clase
        class_metrics = {}
        for i, class_name in enumerate(self.mlb.classes_):
            class_metrics[class_name] = {
                'f1': f1_score(y_test[:, i], y_pred[:, i]),
                'precision': precision_score(y_test[:, i], y_pred[:, i]),
                'recall': recall_score(y_test[:, i], y_pred[:, i]),
                'accuracy': accuracy_score(y_test[:, i], y_pred[:, i])
            }
        
        metrics['class_metrics'] = class_metrics
        self.is_trained = True
        
        print("✅ Entrenamiento completado!")
        print(f"F1-Score Macro: {metrics['f1_macro']:.3f}")
        print(f"F1-Score Micro: {metrics['f1_micro']:.3f}")
        print(f"Tiempo total: {metrics['total_time']:.2f} segundos")
        
        return metrics
    
    def predict(self, title, abstract):
        """
        Predice las categorías de un artículo
        
        Args:
            title (str): Título del artículo
            abstract (str): Abstract del artículo
        
        Returns:
            list: Lista de categorías predichas
        """
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado antes de hacer predicciones")
        
        # Preprocesar texto
        processed_text = self._preprocess_text(title, abstract)
        
        # Vectorizar
        text_vector = self.vectorizer.transform([processed_text])
        
        # Predecir
        prediction = self.model.predict(text_vector)
        
        # Convertir a etiquetas
        categories = self.mlb.inverse_transform(prediction)[0]
        
        return list(categories) if categories else ['No category predicted']
    
    def predict_batch(self, titles, abstracts):
        """
        Predice categorías para múltiples artículos
        
        Args:
            titles (list): Lista de títulos
            abstracts (list): Lista de abstracts
        
        Returns:
            list: Lista de listas con categorías predichas
        """
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado antes de hacer predicciones")
        
        if len(titles) != len(abstracts):
            raise ValueError("Las listas de títulos y abstracts deben tener la misma longitud")
        
        # Preprocesar textos
        processed_texts = []
        for title, abstract in zip(titles, abstracts):
            processed_text = self._preprocess_text(title, abstract)
            processed_texts.append(processed_text)
        
        # Vectorizar
        text_vectors = self.vectorizer.transform(processed_texts)
        
        # Predecir
        predictions = self.model.predict(text_vectors)
        
        # Convertir a etiquetas
        results = []
        for prediction in predictions:
            categories = self.mlb.inverse_transform([prediction])[0]
            results.append(list(categories) if categories else ['No category predicted'])
        
        return results
    
    def save_model(self, file_path):
        """
        Guarda el modelo entrenado
        
        Args:
            file_path (str): Ruta donde guardar el modelo
        """
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado antes de guardarlo")
        
        model_data = {
            'vectorizer': self.vectorizer,
            'model': self.model,
            'mlb': self.mlb,
            'parameters': {
                'max_features': self.max_features,
                'min_df': self.min_df,
                'max_df': self.max_df,
                'ngram_range': self.ngram_range,
                'C': self.C,
                'kernel': self.kernel,
                'random_state': self.random_state
            }
        }
        
        joblib.dump(model_data, file_path)
        print(f"✅ Modelo guardado en: {file_path}")
    
    def load_model(self, file_path):
        """
        Carga un modelo previamente guardado
        
        Args:
            file_path (str): Ruta del modelo guardado
        """
        model_data = joblib.load(file_path)
        
        self.vectorizer = model_data['vectorizer']
        self.model = model_data['model']
        self.mlb = model_data['mlb']
        
        # Cargar parámetros
        params = model_data['parameters']
        self.max_features = params['max_features']
        self.min_df = params['min_df']
        self.max_df = params['max_df']
        self.ngram_range = params['ngram_range']
        self.C = params['C']
        self.kernel = params['kernel']
        self.random_state = params['random_state']
        
        self.is_trained = True
        print(f"✅ Modelo cargado desde: {file_path}")
    
    def get_feature_importance(self, top_n=20):
        """
        Obtiene las características más importantes del modelo
        
        Args:
            top_n (int): Número de características top a retornar
        
        Returns:
            dict: Características importantes por categoría
        """
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado antes de obtener importancia de características")
        
        feature_names = self.vectorizer.get_feature_names_out()
        feature_importance = {}
        
        for i, class_name in enumerate(self.mlb.classes_):
            # Obtener coeficientes del SVM para esta clase
            coef = self.model.estimators_[i].coef_[0]
            
            # Obtener top características positivas y negativas
            top_positive_idx = np.argsort(coef)[-top_n:][::-1]
            top_negative_idx = np.argsort(coef)[:top_n]
            
            feature_importance[class_name] = {
                'positive': [(feature_names[idx], coef[idx]) for idx in top_positive_idx],
                'negative': [(feature_names[idx], coef[idx]) for idx in top_negative_idx]
            }
        
        return feature_importance

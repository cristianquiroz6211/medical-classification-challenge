"""
Funciones de evaluación para modelos de clasificación médica
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, hamming_loss,
    f1_score, precision_score, recall_score, accuracy_score,
    multilabel_confusion_matrix
)


def evaluate_multilabel_model(y_true, y_pred, class_names):
    """
    Evalúa un modelo multi-label y retorna métricas detalladas
    
    Args:
        y_true (array): Etiquetas verdaderas
        y_pred (array): Etiquetas predichas
        class_names (list): Nombres de las clases
    
    Returns:
        dict: Diccionario con métricas de evaluación
    """
    metrics = {}
    
    # Métricas generales
    metrics['hamming_loss'] = hamming_loss(y_true, y_pred)
    metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro')
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
    metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro')
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro')
    metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro')
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro')
    
    # Métricas por clase
    class_metrics = {}
    for i, class_name in enumerate(class_names):
        y_true_class = y_true[:, i]
        y_pred_class = y_pred[:, i]
        
        class_metrics[class_name] = {
            'f1': f1_score(y_true_class, y_pred_class),
            'precision': precision_score(y_true_class, y_pred_class),
            'recall': recall_score(y_true_class, y_pred_class),
            'accuracy': accuracy_score(y_true_class, y_pred_class)
        }
    
    metrics['class_metrics'] = class_metrics
    
    return metrics


def plot_evaluation_metrics(metrics, save_path=None):
    """
    Visualiza las métricas de evaluación
    
    Args:
        metrics (dict): Métricas de evaluación
        save_path (str, optional): Ruta para guardar la figura
    """
    class_metrics = metrics['class_metrics']
    classes = list(class_metrics.keys())
    
    # Extraer métricas por clase
    f1_scores = [class_metrics[cls]['f1'] for cls in classes]
    precisions = [class_metrics[cls]['precision'] for cls in classes]
    recalls = [class_metrics[cls]['recall'] for cls in classes]
    accuracies = [class_metrics[cls]['accuracy'] for cls in classes]
    
    # Crear visualización
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    # F1-Score por clase
    bars1 = axes[0, 0].bar(classes, f1_scores, color=colors[:len(classes)], alpha=0.8)
    axes[0, 0].set_title('F1-Score por Categoría Médica', fontweight='bold', fontsize=14)
    axes[0, 0].set_ylabel('F1-Score')
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Añadir valores en las barras
    for bar, score in zip(bars1, f1_scores):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                       f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Precision por clase
    bars2 = axes[0, 1].bar(classes, precisions, color=colors[:len(classes)], alpha=0.8)
    axes[0, 1].set_title('Precision por Categoría Médica', fontweight='bold', fontsize=14)
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    for bar, score in zip(bars2, precisions):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                       f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Recall por clase
    bars3 = axes[1, 0].bar(classes, recalls, color=colors[:len(classes)], alpha=0.8)
    axes[1, 0].set_title('Recall por Categoría Médica', fontweight='bold', fontsize=14)
    axes[1, 0].set_ylabel('Recall')
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    for bar, score in zip(bars3, recalls):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                       f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Comparación de métricas
    x = np.arange(len(classes))
    width = 0.2
    
    axes[1, 1].bar(x - width, f1_scores, width, label='F1-Score', 
                   color='#FF6B6B', alpha=0.8)
    axes[1, 1].bar(x, precisions, width, label='Precision', 
                   color='#4ECDC4', alpha=0.8)
    axes[1, 1].bar(x + width, recalls, width, label='Recall', 
                   color='#45B7D1', alpha=0.8)
    
    axes[1, 1].set_title('Comparación de Métricas por Categoría', fontweight='bold', fontsize=14)
    axes[1, 1].set_ylabel('Puntuación')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(classes, rotation=45)
    axes[1, 1].legend()
    axes[1, 1].set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Gráfico guardado en: {save_path}")
    
    plt.show()


def print_evaluation_report(metrics):
    """
    Imprime un reporte detallado de evaluación
    
    Args:
        metrics (dict): Métricas de evaluación
    """
    print("📈 REPORTE DE EVALUACIÓN DETALLADO")
    print("="*60)
    
    # Métricas generales
    print("\n🎯 MÉTRICAS GENERALES")
    print("-" * 30)
    print(f"Hamming Loss (menor es mejor):     {metrics['hamming_loss']:.4f}")
    print(f"F1-Score Micro:                    {metrics['f1_micro']:.4f}")
    print(f"F1-Score Macro:                    {metrics['f1_macro']:.4f}")
    print(f"F1-Score Weighted:                 {metrics['f1_weighted']:.4f}")
    print(f"Precision Micro:                   {metrics['precision_micro']:.4f}")
    print(f"Precision Macro:                   {metrics['precision_macro']:.4f}")
    print(f"Recall Micro:                      {metrics['recall_micro']:.4f}")
    print(f"Recall Macro:                      {metrics['recall_macro']:.4f}")
    
    # Métricas por clase
    print("\n🏥 MÉTRICAS POR CATEGORÍA MÉDICA")
    print("-" * 45)
    
    class_metrics = metrics['class_metrics']
    
    # Crear tabla formateada
    header = f"{'Categoría':<15} {'F1':<8} {'Precision':<10} {'Recall':<8} {'Accuracy':<8}"
    print(header)
    print("-" * len(header))
    
    for class_name, class_metric in class_metrics.items():
        row = (f"{class_name:<15} "
               f"{class_metric['f1']:<8.3f} "
               f"{class_metric['precision']:<10.3f} "
               f"{class_metric['recall']:<8.3f} "
               f"{class_metric['accuracy']:<8.3f}")
        print(row)
    
    # Identificar mejor y peor categoría
    f1_scores = {k: v['f1'] for k, v in class_metrics.items()}
    best_category = max(f1_scores, key=f1_scores.get)
    worst_category = min(f1_scores, key=f1_scores.get)
    
    print(f"\n🏆 Mejor categoría (F1): {best_category} ({f1_scores[best_category]:.3f})")
    print(f"🔍 Categoría más desafiante: {worst_category} ({f1_scores[worst_category]:.3f})")


def analyze_prediction_patterns(y_true, y_pred, class_names):
    """
    Analiza patrones de predicción y errores comunes
    
    Args:
        y_true (array): Etiquetas verdaderas
        y_pred (array): Etiquetas predichas
        class_names (list): Nombres de las clases
    
    Returns:
        dict: Análisis de patrones
    """
    patterns = {}
    
    # Matrices de confusión por clase
    conf_matrices = multilabel_confusion_matrix(y_true, y_pred)
    
    patterns['confusion_matrices'] = {}
    for i, class_name in enumerate(class_names):
        tn, fp, fn, tp = conf_matrices[i].ravel()
        patterns['confusion_matrices'][class_name] = {
            'true_negative': tn,
            'false_positive': fp,
            'false_negative': fn,
            'true_positive': tp
        }
    
    # Análisis de co-ocurrencias
    patterns['cooccurrence_analysis'] = analyze_label_cooccurrence(y_true, y_pred, class_names)
    
    return patterns


def analyze_label_cooccurrence(y_true, y_pred, class_names):
    """
    Analiza co-ocurrencias de etiquetas en predicciones vs realidad
    
    Args:
        y_true (array): Etiquetas verdaderas
        y_pred (array): Etiquetas predichas
        class_names (list): Nombres de las clases
    
    Returns:
        dict: Análisis de co-ocurrencias
    """
    # Convertir a DataFrame para facilitar análisis
    df_true = pd.DataFrame(y_true, columns=class_names)
    df_pred = pd.DataFrame(y_pred, columns=class_names)
    
    cooccurrence = {}
    
    # Matriz de co-ocurrencia real
    cooccurrence['true_cooccurrence'] = df_true.T.dot(df_true)
    
    # Matriz de co-ocurrencia predicha
    cooccurrence['pred_cooccurrence'] = df_pred.T.dot(df_pred)
    
    # Diferencia entre real y predicha
    cooccurrence['difference'] = (cooccurrence['true_cooccurrence'] - 
                                 cooccurrence['pred_cooccurrence'])
    
    return cooccurrence


def plot_confusion_matrices(patterns, class_names, save_path=None):
    """
    Visualiza las matrices de confusión por clase
    
    Args:
        patterns (dict): Patrones de predicción
        class_names (list): Nombres de las clases
        save_path (str, optional): Ruta para guardar la figura
    """
    n_classes = len(class_names)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for i, class_name in enumerate(class_names):
        if i < len(axes):
            cm_data = patterns['confusion_matrices'][class_name]
            
            # Crear matriz de confusión 2x2
            cm = np.array([[cm_data['true_negative'], cm_data['false_positive']],
                          [cm_data['false_negative'], cm_data['true_positive']]])
            
            # Plotear matriz de confusión
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['No', 'Sí'], yticklabels=['No', 'Sí'],
                       ax=axes[i])
            axes[i].set_title(f'Matriz de Confusión: {class_name}', fontweight='bold')
            axes[i].set_xlabel('Predicho')
            axes[i].set_ylabel('Real')
    
    # Ocultar ejes no utilizados
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Matrices de confusión guardadas en: {save_path}")
    
    plt.show()


def generate_evaluation_summary(metrics, patterns, model_info=None):
    """
    Genera un resumen completo de la evaluación
    
    Args:
        metrics (dict): Métricas de evaluación
        patterns (dict): Patrones de predicción
        model_info (dict, optional): Información del modelo
    
    Returns:
        str: Resumen de evaluación
    """
    summary = []
    summary.append("📊 RESUMEN EJECUTIVO DE EVALUACIÓN")
    summary.append("="*50)
    
    if model_info:
        summary.append(f"\n🤖 INFORMACIÓN DEL MODELO:")
        summary.append(f"Tipo: {model_info.get('type', 'N/A')}")
        summary.append(f"Características: {model_info.get('features', 'N/A'):,}")
        summary.append(f"Tiempo de entrenamiento: {model_info.get('training_time', 'N/A'):.2f}s")
    
    summary.append(f"\n🎯 RENDIMIENTO GENERAL:")
    summary.append(f"F1-Score Macro: {metrics['f1_macro']:.3f}")
    summary.append(f"F1-Score Micro: {metrics['f1_micro']:.3f}")
    summary.append(f"Hamming Loss: {metrics['hamming_loss']:.4f}")
    
    # Mejor y peor categoría
    class_metrics = metrics['class_metrics']
    f1_scores = {k: v['f1'] for k, v in class_metrics.items()}
    best_category = max(f1_scores, key=f1_scores.get)
    worst_category = min(f1_scores, key=f1_scores.get)
    
    summary.append(f"\n🏆 CATEGORÍA MÁS EXITOSA: {best_category} (F1: {f1_scores[best_category]:.3f})")
    summary.append(f"🔍 CATEGORÍA MÁS DESAFIANTE: {worst_category} (F1: {f1_scores[worst_category]:.3f})")
    
    summary.append(f"\n✅ ESTADO: {'MODELO LISTO PARA PRODUCCIÓN' if metrics['f1_macro'] > 0.8 else 'MODELO REQUIERE MEJORAS'}")
    
    return "\n".join(summary)

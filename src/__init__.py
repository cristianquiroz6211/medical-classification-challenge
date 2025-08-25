"""
Paquete para la clasificación de literatura médica
"""

from .models import MedicalTextClassifier
from .utils import (
    preprocess_medical_text, 
    remove_common_stopwords, 
    load_and_preprocess_data,
    get_dataset_statistics
)
from .evaluation import (
    evaluate_multilabel_model,
    plot_evaluation_metrics,
    print_evaluation_report,
    generate_evaluation_summary
)

__version__ = "1.0.0"
__author__ = "Challenge de Clasificación Médica"

__all__ = [
    'MedicalTextClassifier',
    'preprocess_medical_text',
    'remove_common_stopwords',
    'load_and_preprocess_data',
    'get_dataset_statistics',
    'evaluate_multilabel_model',
    'plot_evaluation_metrics',
    'print_evaluation_report',
    'generate_evaluation_summary'
]

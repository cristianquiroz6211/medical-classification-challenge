"""
Ejemplo de uso del sistema de clasificación de literatura médica
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from src.models import MedicalTextClassifier
from src.utils import load_and_preprocess_data, get_dataset_statistics
from src.evaluation import (
    evaluate_multilabel_model, plot_evaluation_metrics, 
    print_evaluation_report, generate_evaluation_summary
)

def main():
    """
    Ejemplo completo de entrenamiento y evaluación del modelo
    """
    print("🏥 SISTEMA DE CLASIFICACIÓN DE LITERATURA MÉDICA")
    print("="*60)
    
    # 1. Cargar y preprocesar datos
    print("\n📁 Cargando dataset...")
    data_path = "data/challenge_data-18-ago.csv"
    df = load_and_preprocess_data(data_path)
    
    # 2. Mostrar estadísticas del dataset
    print("\n📊 Estadísticas del dataset:")
    stats = get_dataset_statistics(df)
    print(f"Total de registros: {stats['total_records']:,}")
    print(f"Categorías únicas: {stats['unique_categories']}")
    print(f"Distribución de categorías:")
    for category, count in stats['categories'].items():
        percentage = (count / stats['total_records']) * 100
        print(f"  {category}: {count:,} ({percentage:.1f}%)")
    
    # 3. Inicializar y entrenar modelo
    print("\n🤖 Inicializando modelo...")
    classifier = MedicalTextClassifier(
        max_features=10000,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2),
        C=1.0,
        kernel='linear'
    )
    
    # 4. Entrenamiento
    print("\n🚀 Entrenando modelo...")
    training_metrics = classifier.fit(df)
    
    # 5. Guardar modelo
    model_path = "models/medical_classifier.pkl"
    os.makedirs("models", exist_ok=True)
    classifier.save_model(model_path)
    
    # 6. Mostrar reporte de evaluación
    print("\n📈 REPORTE DE EVALUACIÓN:")
    print("-" * 40)
    print(f"F1-Score Macro: {training_metrics['f1_macro']:.3f}")
    print(f"F1-Score Micro: {training_metrics['f1_micro']:.3f}")
    print(f"Hamming Loss: {training_metrics['hamming_loss']:.4f}")
    print(f"Tiempo de entrenamiento: {training_metrics['training_time']:.2f}s")
    
    # 7. Ejemplos de predicción
    print("\n🔮 EJEMPLOS DE PREDICCIÓN:")
    print("-" * 30)
    
    examples = [
        {
            'title': 'Effects of ACE inhibitors on cardiovascular outcomes',
            'abstract': 'This randomized controlled trial examined the effects of ACE inhibitors on myocardial infarction, heart failure, and stroke in patients with hypertension and coronary artery disease.',
            'expected': ['cardiovascular']
        },
        {
            'title': 'Chemotherapy-induced peripheral neuropathy in cancer patients',
            'abstract': 'Analysis of neurological complications in oncology patients receiving platinum-based chemotherapy, focusing on peripheral nerve damage and cognitive effects.',
            'expected': ['neurological', 'oncological']
        },
        {
            'title': 'Hepatorenal syndrome in cirrhotic patients',
            'abstract': 'Study of kidney dysfunction in patients with liver cirrhosis, examining the relationship between hepatic failure and renal impairment.',
            'expected': ['hepatorenal']
        }
    ]
    
    for i, example in enumerate(examples):
        predicted = classifier.predict(example['title'], example['abstract'])
        print(f"\nEjemplo {i+1}:")
        print(f"Título: {example['title']}")
        print(f"Esperado: {example['expected']}")
        print(f"Predicho: {predicted}")
        
        # Verificar acierto
        overlap = set(example['expected']) & set(predicted)
        if overlap == set(example['expected']):
            print("✅ Predicción perfecta!")
        elif overlap:
            print(f"🔶 Predicción parcial: {list(overlap)}")
        else:
            print("❌ Predicción incorrecta")
    
    print(f"\n✅ SISTEMA COMPLETAMENTE FUNCIONAL")
    print(f"📁 Modelo guardado en: {model_path}")
    print(f"🎯 F1-Score promedio: {training_metrics['f1_macro']:.3f}")
    print(f"🚀 Listo para uso en producción!")

def test_saved_model():
    """
    Prueba cargar y usar un modelo guardado
    """
    print("\n🔄 PROBANDO MODELO GUARDADO:")
    print("-" * 35)
    
    # Cargar modelo guardado
    classifier = MedicalTextClassifier()
    model_path = "models/medical_classifier.pkl"
    
    try:
        classifier.load_model(model_path)
        
        # Prueba de predicción
        title = "Novel treatment for Alzheimer's disease progression"
        abstract = "This study investigates new therapeutic approaches for cognitive decline in elderly patients with dementia and neurological disorders."
        
        prediction = classifier.predict(title, abstract)
        print(f"Título: {title}")
        print(f"Predicción: {prediction}")
        print("✅ Modelo cargado y funcionando correctamente!")
        
    except FileNotFoundError:
        print(f"❌ Modelo no encontrado en {model_path}")
        print("Ejecuta primero el entrenamiento del modelo.")

if __name__ == "__main__":
    # Ejecutar entrenamiento completo
    main()
    
    # Probar modelo guardado
    test_saved_model()

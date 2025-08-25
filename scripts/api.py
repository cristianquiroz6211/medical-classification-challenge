# API REST para el Sistema de Clasificación de Literatura Médica

from flask import Flask, request, jsonify
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models import MedicalTextClassifier
import traceback

app = Flask(__name__)

# Cargar modelo al iniciar la aplicación
classifier = None

def load_model():
    global classifier
    try:
        classifier = MedicalTextClassifier()
        model_path = "../models/medical_classifier.pkl"
        classifier.load_model(model_path)
        print("✅ Modelo cargado exitosamente")
        return True
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de verificación de salud"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': classifier is not None and classifier.is_trained
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint para predecir categorías de un artículo médico
    
    Ejemplo de payload:
    {
        "title": "Effects of statins on cardiovascular outcomes",
        "abstract": "This study examines the cardiovascular protective effects..."
    }
    """
    try:
        # Verificar que el modelo esté cargado
        if not classifier or not classifier.is_trained:
            return jsonify({
                'error': 'Modelo no disponible',
                'message': 'El modelo no ha sido cargado correctamente'
            }), 500
        
        # Obtener datos del request
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'Datos inválidos',
                'message': 'No se recibieron datos JSON'
            }), 400
        
        title = data.get('title', '')
        abstract = data.get('abstract', '')
        
        if not title and not abstract:
            return jsonify({
                'error': 'Datos incompletos',
                'message': 'Se requiere al menos título o abstract'
            }), 400
        
        # Realizar predicción
        categories = classifier.predict(title, abstract)
        
        return jsonify({
            'success': True,
            'predictions': categories,
            'input': {
                'title': title,
                'abstract': abstract
            }
        })
        
    except Exception as e:
        return jsonify({
            'error': 'Error de predicción',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Endpoint para predecir categorías de múltiples artículos
    
    Ejemplo de payload:
    {
        "articles": [
            {
                "title": "Título 1",
                "abstract": "Abstract 1"
            },
            {
                "title": "Título 2", 
                "abstract": "Abstract 2"
            }
        ]
    }
    """
    try:
        # Verificar que el modelo esté cargado
        if not classifier or not classifier.is_trained:
            return jsonify({
                'error': 'Modelo no disponible',
                'message': 'El modelo no ha sido cargado correctamente'
            }), 500
        
        # Obtener datos del request
        data = request.get_json()
        
        if not data or 'articles' not in data:
            return jsonify({
                'error': 'Datos inválidos',
                'message': 'Se requiere un array de artículos'
            }), 400
        
        articles = data['articles']
        
        if not isinstance(articles, list) or len(articles) == 0:
            return jsonify({
                'error': 'Datos inválidos',
                'message': 'El campo articles debe ser un array no vacío'
            }), 400
        
        # Extraer títulos y abstracts
        titles = []
        abstracts = []
        
        for i, article in enumerate(articles):
            if not isinstance(article, dict):
                return jsonify({
                    'error': 'Datos inválidos',
                    'message': f'El artículo {i} debe ser un objeto JSON'
                }), 400
            
            titles.append(article.get('title', ''))
            abstracts.append(article.get('abstract', ''))
        
        # Realizar predicciones
        predictions = classifier.predict_batch(titles, abstracts)
        
        # Formatear respuesta
        results = []
        for i, (title, abstract, categories) in enumerate(zip(titles, abstracts, predictions)):
            results.append({
                'index': i,
                'title': title,
                'abstract': abstract,
                'predictions': categories
            })
        
        return jsonify({
            'success': True,
            'results': results,
            'total_processed': len(results)
        })
        
    except Exception as e:
        return jsonify({
            'error': 'Error de predicción',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/categories', methods=['GET'])
def get_categories():
    """Obtiene las categorías soportadas por el modelo"""
    try:
        if not classifier or not classifier.is_trained:
            return jsonify({
                'error': 'Modelo no disponible',
                'message': 'El modelo no ha sido cargado correctamente'
            }), 500
        
        categories = list(classifier.mlb.classes_)
        
        return jsonify({
            'success': True,
            'categories': categories,
            'total_categories': len(categories)
        })
        
    except Exception as e:
        return jsonify({
            'error': 'Error obteniendo categorías',
            'message': str(e)
        }), 500

@app.route('/model_info', methods=['GET'])
def get_model_info():
    """Obtiene información del modelo cargado"""
    try:
        if not classifier or not classifier.is_trained:
            return jsonify({
                'error': 'Modelo no disponible',
                'message': 'El modelo no ha sido cargado correctamente'
            }), 500
        
        info = {
            'model_type': 'SVM Multi-Label',
            'vectorizer_type': 'TF-IDF',
            'max_features': classifier.max_features,
            'ngram_range': classifier.ngram_range,
            'categories': list(classifier.mlb.classes_),
            'total_features': len(classifier.vectorizer.get_feature_names_out()) if classifier.vectorizer else 0,
            'kernel': classifier.kernel,
            'C_parameter': classifier.C
        }
        
        return jsonify({
            'success': True,
            'model_info': info
        })
        
    except Exception as e:
        return jsonify({
            'error': 'Error obteniendo información del modelo',
            'message': str(e)
        }), 500

@app.route('/', methods=['GET'])
def home():
    """Página de inicio con documentación de la API"""
    documentation = {
        'title': 'API de Clasificación de Literatura Médica',
        'description': 'Sistema de IA para clasificar artículos médicos en categorías especializadas',
        'version': '1.0.0',
        'endpoints': {
            'GET /health': 'Verificación de salud del servicio',
            'GET /categories': 'Obtener categorías soportadas',
            'GET /model_info': 'Información del modelo',
            'POST /predict': 'Predecir categorías de un artículo',
            'POST /predict_batch': 'Predecir categorías de múltiples artículos'
        },
        'categories': [
            'cardiovascular',
            'neurological', 
            'hepatorenal',
            'oncological'
        ],
        'example_usage': {
            'predict': {
                'url': '/predict',
                'method': 'POST',
                'payload': {
                    'title': 'Effects of ACE inhibitors on myocardial infarction',
                    'abstract': 'This study examines cardiovascular outcomes...'
                }
            }
        }
    }
    
    return jsonify(documentation)

if __name__ == '__main__':
    print("🚀 Iniciando API de Clasificación de Literatura Médica...")
    
    # Cargar modelo
    if load_model():
        print("🌐 Servidor iniciando en http://localhost:5000")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("❌ No se pudo cargar el modelo. Verifica que existe el archivo models/medical_classifier.pkl")
        print("💡 Ejecuta main.py primero para entrenar y guardar el modelo.")

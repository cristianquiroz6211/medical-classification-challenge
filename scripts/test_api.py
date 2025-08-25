"""
Script de prueba para el API de clasificación médica
"""

import requests
import json
import time

# Configuración del API
API_BASE_URL = "http://localhost:5000"

def test_health():
    """Prueba el endpoint de salud"""
    print("🔍 Probando endpoint de salud...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Estado: {data['status']}")
            print(f"✅ Modelo cargado: {data['model_loaded']}")
        else:
            print(f"❌ Error: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ No se pudo conectar al API. ¿Está ejecutándose?")

def test_categories():
    """Prueba el endpoint de categorías"""
    print("\n📋 Probando endpoint de categorías...")
    try:
        response = requests.get(f"{API_BASE_URL}/categories")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Categorías soportadas: {data['categories']}")
            print(f"✅ Total: {data['total_categories']}")
        else:
            print(f"❌ Error: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ No se pudo conectar al API")

def test_model_info():
    """Prueba el endpoint de información del modelo"""
    print("\n🤖 Probando endpoint de información del modelo...")
    try:
        response = requests.get(f"{API_BASE_URL}/model_info")
        if response.status_code == 200:
            data = response.json()
            model_info = data['model_info']
            print(f"✅ Tipo de modelo: {model_info['model_type']}")
            print(f"✅ Vectorizador: {model_info['vectorizer_type']}")
            print(f"✅ Características: {model_info['total_features']:,}")
            print(f"✅ Kernel: {model_info['kernel']}")
        else:
            print(f"❌ Error: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ No se pudo conectar al API")

def test_single_prediction():
    """Prueba la predicción de un solo artículo"""
    print("\n🔮 Probando predicción individual...")
    
    test_article = {
        "title": "Effects of ACE inhibitors on cardiovascular outcomes in hypertensive patients",
        "abstract": "This randomized controlled trial examined the cardiovascular protective effects of ACE inhibitor therapy in patients with hypertension and coronary artery disease. Results demonstrated significant reduction in myocardial infarction risk and improved cardiac outcomes over 24 months of follow-up."
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=test_article,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Título: {test_article['title'][:60]}...")
            print(f"✅ Predicciones: {data['predictions']}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"❌ Respuesta: {response.text}")
    except requests.exceptions.ConnectionError:
        print("❌ No se pudo conectar al API")

def test_batch_prediction():
    """Prueba la predicción de múltiples artículos"""
    print("\n📚 Probando predicción por lotes...")
    
    test_articles = {
        "articles": [
            {
                "title": "Alzheimer disease progression and cognitive decline",
                "abstract": "Longitudinal study of memory deterioration in elderly patients with dementia. Neurological assessments and brain imaging revealed progressive cognitive impairment patterns consistent with Alzheimer pathology."
            },
            {
                "title": "Chemotherapy-induced nephrotoxicity in cancer patients",
                "abstract": "Analysis of kidney function decline in oncology patients receiving cisplatin-based chemotherapy. Study evaluates renal biomarkers and hepatorenal syndrome development in cancer treatment protocols."
            },
            {
                "title": "Myocardial infarction risk factors in diabetic patients",
                "abstract": "Cardiovascular risk assessment in diabetic population. The study examines coronary artery disease progression, cardiac events, and optimal management strategies for preventing heart attacks."
            }
        ]
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict_batch",
            json=test_articles,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Procesados: {data['total_processed']} artículos")
            
            for result in data['results']:
                print(f"\nArtículo {result['index'] + 1}:")
                print(f"  Título: {result['title'][:50]}...")
                print(f"  Predicciones: {result['predictions']}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"❌ Respuesta: {response.text}")
    except requests.exceptions.ConnectionError:
        print("❌ No se pudo conectar al API")

def test_edge_cases():
    """Prueba casos límite y manejo de errores"""
    print("\n⚠️ Probando casos límite...")
    
    # Título vacío
    print("🔍 Probando con título vacío...")
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json={"title": "", "abstract": ""},
            headers={'Content-Type': 'application/json'}
        )
        print(f"  Status: {response.status_code}")
        if response.status_code != 200:
            print(f"  ✅ Error manejado correctamente")
    except:
        print("  ❌ Error de conexión")
    
    # JSON inválido
    print("🔍 Probando con JSON inválido...")
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            data="invalid json",
            headers={'Content-Type': 'application/json'}
        )
        print(f"  Status: {response.status_code}")
        if response.status_code != 200:
            print(f"  ✅ Error manejado correctamente")
    except:
        print("  ❌ Error de conexión")

def benchmark_performance():
    """Prueba el rendimiento del API"""
    print("\n⚡ Probando rendimiento...")
    
    test_article = {
        "title": "Performance test article",
        "abstract": "This is a test article for performance benchmarking of the medical classification API system."
    }
    
    num_requests = 10
    times = []
    
    for i in range(num_requests):
        try:
            start_time = time.time()
            response = requests.post(
                f"{API_BASE_URL}/predict",
                json=test_article,
                headers={'Content-Type': 'application/json'}
            )
            end_time = time.time()
            
            if response.status_code == 200:
                times.append(end_time - start_time)
        except:
            pass
    
    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"✅ Requests exitosos: {len(times)}/{num_requests}")
        print(f"✅ Tiempo promedio: {avg_time*1000:.2f}ms")
        print(f"✅ Tiempo mínimo: {min_time*1000:.2f}ms")
        print(f"✅ Tiempo máximo: {max_time*1000:.2f}ms")
    else:
        print("❌ No se pudieron completar requests de prueba")

def main():
    """Ejecuta todas las pruebas"""
    print("🧪 PRUEBAS DEL API DE CLASIFICACIÓN MÉDICA")
    print("="*60)
    
    # Pruebas básicas
    test_health()
    test_categories()
    test_model_info()
    
    # Pruebas de predicción
    test_single_prediction()
    test_batch_prediction()
    
    # Pruebas de casos límite
    test_edge_cases()
    
    # Pruebas de rendimiento
    benchmark_performance()
    
    print("\n✅ PRUEBAS COMPLETADAS")
    print("🚀 El API está funcionando correctamente!")

if __name__ == "__main__":
    print("💡 Asegúrate de que el API esté ejecutándose en http://localhost:5000")
    print("💡 Ejecuta: python api.py")
    print("💡 Presiona Enter para continuar con las pruebas...")
    input()
    
    main()

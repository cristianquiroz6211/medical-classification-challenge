# 🚀 Guía de Despliegue - Medical Classification Challenge

## 📋 Resumen del Sistema

Sistema completo de clasificación automática de literatura médica utilizando Machine Learning. Implementa clasificación multi-label para 4 categorías médicas principales.

### 🎯 Métricas Alcanzadas
- **F1-Score Macro**: 0.864
- **F1-Score Micro**: 0.878  
- **Hamming Loss**: 0.076
- **Precisión Micro**: 0.946

## 🏗️ Arquitectura del Sistema

```
medical-classification-challenge/
├── 📁 data/                 # Dataset (3,565 artículos médicos)
├── 📁 src/                  # Código fuente principal
│   ├── models.py            # Clasificador TF-IDF + SVM
│   ├── utils.py             # Preprocesamiento médico
│   └── evaluation.py        # Métricas y evaluación
├── 📁 scripts/              # Scripts ejecutables
│   ├── api.py              # API REST Flask
│   ├── test_api.py         # Pruebas del API
│   └── generate_report.py  # Generador de reportes
├── 📁 notebooks/           # Análisis exploratorio
├── 📁 models/              # Modelos entrenados
├── 📁 results/             # Resultados y visualizaciones
└── 📄 main.py              # Script principal de entrenamiento
```

## 🚀 Despliegue Rápido

### 1. Configuración del Entorno

```bash
# Clonar repositorio
git clone <repository-url>
cd medical-classification-challenge

# Crear entorno virtual
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Entrenamiento del Modelo

```bash
# Entrenar modelo completo
python main.py

# Output esperado:
# ✅ Modelo entrenado exitosamente
# 📊 F1-Score Macro: 0.864
# 📊 F1-Score Micro: 0.878
# 💾 Modelo guardado en: models/medical_classifier.pkl
```

### 3. Lanzamiento del API

```bash
# Iniciar servidor Flask
cd scripts/
python api.py

# Servidor corriendo en: http://localhost:5000
```

### 4. Pruebas del Sistema

```bash
# Ejecutar pruebas automáticas
python scripts/test_api.py

# Output esperado:
# ✅ Todas las pruebas pasaron
# 🌐 API funcionando correctamente
```

## 🔧 Configuración Avanzada

### Variables de Entorno

```bash
# Opcional: Configurar puerto del API
export FLASK_PORT=5000
export FLASK_DEBUG=False
```

### Configuración de Producción

```python
# En scripts/api.py - Para producción
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

## 📡 Endpoints del API

### Predicción Individual
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Cardiovascular risk factors in diabetes",
    "abstract": "This study examines the relationship between diabetes and heart disease..."
  }'
```

### Predicción por Lotes
```bash
curl -X POST http://localhost:5000/predict_batch \
  -H "Content-Type: application/json" \
  -d '{
    "articles": [
      {"title": "Heart disease study", "abstract": "Cardiovascular research..."},
      {"title": "Brain tumor analysis", "abstract": "Neurological investigation..."}
    ]
  }'
```

### Endpoints de Sistema
```bash
# Estado de salud
curl http://localhost:5000/health

# Categorías soportadas
curl http://localhost:5000/categories

# Información del modelo
curl http://localhost:5000/model_info
```

## 🐳 Despliegue con Docker

### Dockerfile

```dockerfile
FROM python:3.8-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "scripts/api.py"]
```

### Docker Compose

```yaml
version: '3.8'
services:
  medical-classifier:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./models:/app/models
    environment:
      - FLASK_PORT=5000
```

## ☁️ Despliegue en la Nube

### Heroku

```bash
# Crear Procfile
echo "web: python scripts/api.py" > Procfile

# Desplegar
heroku create medical-classifier-app
git push heroku main
```

### AWS Lambda

```python
# Usar Zappa para deployment serverless
pip install zappa
zappa init
zappa deploy production
```

## 🔍 Monitoreo y Logs

### Configuración de Logs

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
```

### Métricas de Sistema

```bash
# Monitorear uso de memoria
python -c "from src.models import MedicalTextClassifier; import psutil; print(f'RAM: {psutil.virtual_memory().percent}%')"

# Tiempo de respuesta del API
curl -w "@curl-format.txt" -s -o /dev/null http://localhost:5000/health
```

## 🚨 Troubleshooting

### Problemas Comunes

1. **Error de importación**
   ```bash
   # Verificar estructura de directorios
   python -c "import src.models; print('✅ Import OK')"
   ```

2. **Modelo no encontrado**
   ```bash
   # Verificar que el modelo existe
   ls -la models/medical_classifier.pkl
   ```

3. **Puerto ocupado**
   ```bash
   # Cambiar puerto en scripts/api.py
   app.run(host='0.0.0.0', port=5001)
   ```

4. **Memoria insuficiente**
   ```bash
   # Optimizar memoria en producción
   export PYTHONHASHSEED=0
   ```

## 📚 Documentación Adicional

- **API Documentation**: `/docs` endpoint (Swagger UI)
- **Model Performance**: `results/reporte_final.md`
- **Development Guide**: `notebooks/exploracion_dataset.ipynb`
- **Testing Guide**: `scripts/test_api.py`

## 🔐 Seguridad

### Autenticación (Opcional)

```python
# Agregar API key simple
@app.before_request
def require_api_key():
    if request.endpoint != 'health':
        api_key = request.headers.get('X-API-Key')
        if not api_key or api_key != 'your-secret-key':
            return jsonify({'error': 'Invalid API key'}), 401
```

### Rate Limiting

```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["100 per hour"]
)
```

## 📞 Soporte

Para problemas técnicos o preguntas:

- 📧 **Email**: [tu-email@ejemplo.com]
- 🐙 **GitHub Issues**: [repository-url]/issues
- 📖 **Documentation**: [repository-url]/wiki

---

<div align="center">

**🎯 Sistema listo para producción con alta precisión en clasificación médica**

[![Deployment Status](https://img.shields.io/badge/deployment-ready-green.svg)](.)
[![API Status](https://img.shields.io/badge/API-operational-blue.svg)](.)

</div>

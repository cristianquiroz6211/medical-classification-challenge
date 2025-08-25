# 🏥 Challenge de Clasificación de Literatura Médica

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![F1-Score](https://img.shields.io/badge/F1--Score-0.864-brightgreen.svg)]()
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success.svg)]()

> Sistema de Inteligencia Artificial para la clasificación automática de artículos médicos en dominios especializados basándose únicamente en el título y el abstract.

## 🎯 Descripción del Proyecto

Este proyecto implementa una solución completa de machine learning para clasificar automáticamente literatura médica en las siguientes categorías:

- **💓 Cardiovascular**: Enfermedades del corazón y sistema circulatorio
- **🧠 Neurological**: Trastornos del sistema nervioso
- **🩺 Hepatorenal**: Afecciones hepáticas y renales  
- **🎗️ Oncological**: Cáncer y tratamientos oncológicos

## 📊 Resultados Alcanzados

### 🏆 Métricas de Rendimiento
- **F1-Score Macro**: `0.864` ✅
- **F1-Score Micro**: `0.878` ✅  
- **Hamming Loss**: `0.076` ✅
- **Precisión Micro**: `0.946` ✅

### 📈 Rendimiento por Categoría
| Categoría | F1-Score | Precision | Recall | Accuracy |
|-----------|----------|-----------|---------|----------|
| Neurological | 0.909 | 0.908 | 0.911 | 0.909 |
| Cardiovascular | 0.891 | 0.972 | 0.823 | 0.928 |
| Hepatorenal | 0.832 | 0.969 | 0.728 | 0.910 |
| Oncological | 0.824 | 1.000 | 0.700 | 0.950 |

## 🚀 Inicio Rápido

### 📋 Prerrequisitos

```bash
python >= 3.8
pip >= 20.0
```

### ⚡ Instalación

1. **Clonar el repositorio**
```bash
git clone https://github.com/cristianquiroz6211/medical-classification-challenge.git
cd medical-classification-challenge
```

2. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

3. **Entrenar el modelo**
```bash
python main.py
```

### 🔮 Uso Básico

```python
from src.models import MedicalTextClassifier

# Cargar modelo entrenado
classifier = MedicalTextClassifier()
classifier.load_model("models/medical_classifier.pkl")

# Predecir categorías
title = "Effects of ACE inhibitors on cardiovascular outcomes"
abstract = "This study evaluates the impact of ACE inhibitors on heart disease..."

predictions = classifier.predict(title, abstract)
print(f"Categorías predichas: {predictions}")
# Output: ['cardiovascular']
```

## 🏗️ Estructura del Proyecto

```
medical-classification-challenge/
│
├── 📁 data/                          # Datasets
│   └── challenge_data-18-ago.csv
│
├── 📁 src/                           # Código fuente
│   ├── __init__.py
│   ├── models.py                     # Clasificador principal
│   ├── utils.py                      # Utilidades de preprocesamiento
│   └── evaluation.py                 # Métricas y evaluación
│
├── 📁 notebooks/                     # Jupyter notebooks
│   └── exploracion_dataset.ipynb     # EDA y experimentación
│
├── 📁 scripts/                       # Scripts ejecutables
│   ├── api.py                        # API REST
│   ├── test_api.py                   # Pruebas del API
│   └── generate_report.py            # Generador de reportes
│
├── 📁 models/                        # Modelos entrenados
│   └── medical_classifier.pkl
│
├── 📁 results/                       # Resultados y reportes
│   ├── reporte_final.md
│   └── figures/
│
├── 📄 main.py                        # Script principal
├── 📄 requirements.txt               # Dependencias
├── 📄 README.md                      # Este archivo
├── 📄 LICENSE                        # Licencia MIT
└── 📄 .gitignore                     # Archivos ignorados
```

## 🛠️ Metodología Técnica

### 🤖 Algoritmo Principal
- **Modelo Base**: SVM Multi-Label 
- **Vectorización**: TF-IDF (10,000 características)
- **N-gramas**: Unigrams y bigrams (1-2)
- **Kernel**: Lineal (optimizado para texto)

### 📝 Preprocesamiento Especializado
- Limpieza conservando terminología médica
- Remoción de stopwords generales (no médicas)
- Normalización de texto científico
- Manejo de casos multi-label

### ✅ Justificación del Enfoque

**¿Por qué TF-IDF + SVM?**
- ✅ **Precisión**: Excelente captura de terminología médica específica
- ✅ **Eficiencia**: Entrenamiento rápido (~10 segundos)
- ✅ **Robustez**: Manejo efectivo del desbalance de clases
- ✅ **Interpretabilidad**: Modelo explicable para dominio médico

## 🌐 API REST

### 🚀 Iniciar Servidor

```bash
cd scripts/
python api.py
```

### 📡 Endpoints Disponibles

#### Predicción Individual
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Alzheimer disease progression",
    "abstract": "Study of cognitive decline in elderly patients..."
  }'
```

#### Predicción por Lotes
```bash
curl -X POST http://localhost:5000/predict_batch \
  -H "Content-Type: application/json" \
  -d '{
    "articles": [
      {"title": "Título 1", "abstract": "Abstract 1"},
      {"title": "Título 2", "abstract": "Abstract 2"}
    ]
  }'
```

#### Información del Sistema
```bash
# Salud del servicio
curl http://localhost:5000/health

# Categorías soportadas  
curl http://localhost:5000/categories

# Información del modelo
curl http://localhost:5000/model_info
```

## Resultados

### Métricas de Evaluación Alcanzadas
- **F1-Score Macro**: 0.864
- **F1-Score Micro**: 0.878  
- **Hamming Loss**: 0.076
- **Exactitud por categoría**:
  - Cardiovascular: F1 = 0.891, Precision = 0.972, Recall = 0.823
  - Neurological: F1 = 0.909, Precision = 0.908, Recall = 0.911
  - Hepatorenal: F1 = 0.832, Precision = 0.969, Recall = 0.728
  - Oncological: F1 = 0.824, Precision = 1.000, Recall = 0.700

### Justificación del Enfoque

1. **TF-IDF + SVM**: 
   - ✅ Excelente para capturar patrones léxicos específicos y terminología médica
   - ✅ Robusto con datasets de tamaño mediano
   - ✅ Interpretable y eficiente computacionalmente

2. **Preprocesamiento Especializado**: 
   - ✅ Conserva terminología médica importante
   - ✅ Maneja efectivamente el texto multi-label
   - ✅ Optimizado para literatura científica

3. **Multi-Label Learning**: 
   - ✅ Maneja correctamente artículos con múltiples categorías (30.6% del dataset)
   - ✅ Métricas apropiadas para evaluación multi-label
   - ✅ Predicciones realistas para casos complejos

## Uso Rápido

### Entrenamiento y Evaluación

```python
from src.models import MedicalTextClassifier
from src.utils import load_and_preprocess_data

# Cargar datos
df = load_and_preprocess_data("challenge_data-18-ago.csv")

# Entrenar modelo
classifier = MedicalTextClassifier()
metrics = classifier.fit(df)

# Guardar modelo
classifier.save_model("models/medical_classifier.pkl")
```

### Predicción

```python
# Cargar modelo entrenado
classifier = MedicalTextClassifier()
classifier.load_model("models/medical_classifier.pkl")

# Predecir categorías
title = "Effects of ACE inhibitors on cardiovascular outcomes"
abstract = "This study evaluates the impact of ACE inhibitors..."

predictions = classifier.predict(title, abstract)
print(f"Categorías predichas: {predictions}")
# Output: ['cardiovascular']
```

### API REST

```bash
# Iniciar servidor
python api.py

# Hacer predicción via API
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Alzheimer disease progression",
    "abstract": "Study of cognitive decline in elderly patients..."
  }'
```

## Instalación Completa

### 1. Clonar Repositorio

```bash
git clone https://github.com/tu-usuario/medical-classification-challenge.git
cd medical-classification-challenge
```

### 2. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 3. Entrenar Modelo

```bash
python main.py
```

### 4. Probar API (Opcional)

```bash
# Terminal 1: Iniciar servidor
python scripts/api.py

# Terminal 2: Probar API
python scripts/test_api.py
```

## 📊 Dataset

### 📈 Información General
- **Registros totales**: 3,565 artículos médicos
- **Fuentes**: NCBI, BC5CDR y datos sintéticos  
- **Idioma**: Inglés
- **Tipo**: Multi-label (30.6% con múltiples categorías)

### 📋 Estructura de Datos
| Campo | Descripción | Ejemplo |
|-------|-------------|---------|
| `title` | Título del artículo médico | "Effects of ACE inhibitors..." |
| `abstract` | Resumen científico | "This study examines..." |
| `group` | Categorías objetivo | "cardiovascular\|neurological" |

### 📊 Distribución de Categorías
- **Neurological**: 1,785 artículos (50.1%)
- **Cardiovascular**: 1,268 artículos (35.6%)  
- **Hepatorenal**: 1,091 artículos (30.6%)
- **Oncological**: 601 artículos (16.9%)

## 🧪 Pruebas y Validación

### 🔬 Ejecutar Pruebas

```bash
# Entrenar y validar modelo
python main.py

# Generar reporte completo
python scripts/generate_report.py

# Probar API
python scripts/test_api.py
```

### 📊 Visualizaciones

El proyecto genera automáticamente:
- Gráficos de métricas por categoría
- Matrices de confusión multi-label  
- Distribuciones del dataset
- Análisis de co-ocurrencias

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -m 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

## 📈 Próximos Pasos

- [ ] **Optimización**: Grid Search para hiperparámetros
- [ ] **Deep Learning**: Implementación con BioBERT
- [ ] **Ensemble**: Combinación de múltiples modelos
- [ ] **Active Learning**: Mejora continua con nuevos datos
- [ ] **Ontologías**: Integración con MeSH terms

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver [LICENSE](LICENSE) para más detalles.

## 👨‍💻 Autor

**Challenge de Clasificación de Literatura Médica**
- 📧 Email: [Cristiandavidq7@gmail.com]
- 🐙 GitHub: [(https://github.com/cristianquiroz6211)]
- 💼 LinkedIn: [https://www.linkedin.com/in/cristianquiroz1034916211/]
- Reto de : [https://techspherecolombia.com/ai-data-challenge/]

## 🙏 Agradecimientos

- NCBI por los datos médicos
- BC5CDR dataset
- Comunidad de machine learning médico
- Scikit-learn y Python ecosystem
-

---

<div align="center">

**⭐ Si este proyecto te resulta útil, considera darle una estrella ⭐**

[![GitHub stars](https://img.shields.io/github/stars/cristianquiroz6211/medical-classification-challenge.svg?style=social&label=Star)](https://github.com/cristianquiroz6211/medical-classification-challenge)

</div>

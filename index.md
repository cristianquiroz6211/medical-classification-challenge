---
layout: default
title: Medical Classification Challenge
---

# 🏥 Challenge de Clasificación de Literatura Médica

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![F1-Score](https://img.shields.io/badge/F1--Score-0.864-brightgreen.svg)]()
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success.svg)]()

## 🎯 Acerca del Proyecto

Sistema de **Inteligencia Artificial** para la clasificación automática de artículos médicos utilizando **Machine Learning** avanzado. El sistema analiza únicamente el título y abstract de papers médicos para clasificarlos automáticamente en cuatro dominios especializados.

### 🏆 Resultados Destacados

| Métrica | Valor | Estado |
|---------|-------|--------|
| **F1-Score Macro** | 0.864 | ✅ Excelente |
| **F1-Score Micro** | 0.878 | ✅ Excelente |
| **Precisión Micro** | 0.946 | ✅ Sobresaliente |
| **Hamming Loss** | 0.076 | ✅ Muy Bajo |

### 🔬 Categorías Médicas Soportadas

- **💓 Cardiovascular**: Enfermedades del corazón y sistema circulatorio
- **🧠 Neurológico**: Trastornos del sistema nervioso
- **🩺 Hepatorenal**: Afecciones hepáticas y renales
- **🎗️ Oncológico**: Cáncer y tratamientos oncológicos

## 🚀 Demo en Vivo

### API REST Endpoints

```bash
# Predicción individual
POST /predict
{
  "title": "Effects of ACE inhibitors on cardiovascular outcomes",
  "abstract": "This study evaluates the impact of ACE inhibitors..."
}

# Predicción por lotes
POST /predict_batch
{
  "articles": [
    {"title": "...", "abstract": "..."},
    {"title": "...", "abstract": "..."}
  ]
}
```

### Ejemplo de Respuesta

```json
{
  "predictions": ["cardiovascular"],
  "confidence": {
    "cardiovascular": 0.94,
    "neurological": 0.12,
    "hepatorenal": 0.08,
    "oncological": 0.03
  },
  "processing_time": "0.045s"
}
```

## 📊 Tecnologías Implementadas

### 🤖 Machine Learning
- **Algoritmo**: TF-IDF + SVM Multi-Label
- **Vectorización**: 10,000 características
- **N-gramas**: Unigrams y bigrams
- **Kernel**: Lineal optimizado

### 🌐 Arquitectura del Sistema
- **API**: Flask REST con múltiples endpoints
- **Containerización**: Docker + Docker Compose
- **CI/CD**: GitHub Actions pipeline
- **Testing**: Pruebas automatizadas
- **Documentación**: Completa y actualizada

## 📁 Estructura del Proyecto

```
medical-classification-challenge/
├── 📁 src/                    # Código fuente modular
│   ├── models.py              # Clasificador ML principal
│   ├── utils.py               # Preprocessing médico
│   └── evaluation.py          # Métricas y evaluación
├── 📁 scripts/                # Scripts ejecutables
│   ├── api.py                 # API REST Flask
│   └── test_api.py            # Testing automatizado
├── 📁 data/                   # Dataset (3,565 artículos)
├── 📁 models/                 # Modelos entrenados
├── 📁 notebooks/              # Análisis exploratorio
└── 📁 results/                # Reportes y visualizaciones
```

## 🛠️ Instalación y Uso

### Método 1: Instalación Local

```bash
# Clonar repositorio
git clone https://github.com/cristianquiroz6211/medical-classification-challenge.git
cd medical-classification-challenge

# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt

# Entrenar modelo
python main.py

# Lanzar API
python scripts/api.py
```

### Método 2: Docker

```bash
# Usando Docker Compose
docker-compose up --build

# O usando Docker directamente
docker build -t medical-classifier .
docker run -p 5000:5000 medical-classifier
```

## 📋 Documentación Completa

- **[📖 README Completo](README.md)** - Documentación técnica detallada
- **[🚀 Guía de Despliegue](DEPLOYMENT.md)** - Instrucciones de producción
- **[📊 Análisis Exploratorio](notebooks/exploracion_dataset.ipynb)** - EDA completo
- **[🧪 Reporte Técnico](results/reporte_final.md)** - Resultados y métricas

## 🎯 Casos de Uso

### Para Investigadores
- Clasificación automática de literatura médica
- Análisis de tendencias en publicaciones
- Organización de bases de datos médicas

### Para Instituciones de Salud
- Categorización de protocolos médicos
- Análisis de especialidades médicas
- Gestión de conocimiento médico

### Para Desarrolladores
- API REST lista para integración
- Sistema modular y escalable
- Documentación completa incluida

## 🏆 Resultados por Categoría

<div align="center">

| Categoría | F1-Score | Precision | Recall | Accuracy |
|-----------|:--------:|:---------:|:------:|:--------:|
| **Neurológico** | 0.909 | 0.908 | 0.911 | 90.9% |
| **Cardiovascular** | 0.891 | 0.972 | 0.823 | 92.8% |
| **Hepatorenal** | 0.832 | 0.969 | 0.728 | 91.0% |
| **Oncológico** | 0.824 | 1.000 | 0.700 | 95.0% |

</div>

## 🔗 Enlaces Útiles

- **[🐙 Repositorio GitHub](https://github.com/cristianquiroz6211/medical-classification-challenge)**
- **[📊 Reto Original](https://techspherecolombia.com/ai-data-challenge/)**
- **[👨‍💻 Perfil del Autor](https://github.com/cristianquiroz6211)**
- **[💼 LinkedIn](https://www.linkedin.com/in/cristianquiroz1034916211/)**

## 📞 Contacto y Soporte

¿Tienes preguntas o sugerencias? ¡Contáctame!

- **📧 Email**: cristiandavidq7@gmail.com
- **🐙 GitHub**: [@cristianquiroz6211](https://github.com/cristianquiroz6211)
- **💼 LinkedIn**: [Cristian Quiroz](https://www.linkedin.com/in/cristianquiroz1034916211/)

---

<div align="center">

**⭐ Si este proyecto te resulta útil, ¡considera darle una estrella! ⭐**

[![GitHub stars](https://img.shields.io/github/stars/cristianquiroz6211/medical-classification-challenge.svg?style=social&label=Star)](https://github.com/cristianquiroz6211/medical-classification-challenge)

**🚀 Desarrollado con ❤️ para la comunidad médica y de IA**

</div>

# 🤖 ML Mega Calculator (Web Edition)

> **AutoML Suite for Regression & Classification Tasks**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Framework](https://img.shields.io/badge/Framework-Flask-green)
![Library](https://img.shields.io/badge/ML-Scikit--learn-orange)
![Status](https://img.shields.io/badge/Status-Prototype-yellow)

Una herramienta web de **Machine Learning Automatizado (AutoML)** diseñada para democratizar el acceso a modelos predictivos. Permite a usuarios cargar sus propios datasets, seleccionar variables objetivo y entrenar múltiples algoritmos sin escribir una sola línea de código.

---

## 🚀 Características Principales

* **Carga de Datos Flexible:** Soporte para archivos `.csv` con detección automática de delimitadores.
* **Selección Inteligente de Features:** Interfaz visual para definir variables predictoras ($X$) y variable objetivo ($y$).
* **Detección de Tarea:** Identificación automática de problemas de **Regresión** (valores continuos) o **Clasificación** (categorías).
* **Multi-Algoritmo:**
    * *Regresión:* Linear Regression, SVR, Random Forest Regressor, KNN.
    * *Clasificación:* Logistic Regression, SVM, Random Forest Classifier, Decision Trees.
* **Métricas en Tiempo Real:** Cálculo instantáneo de $R^2$, MSE, RMSE y MAE para regresión; Accuracy y F1-Score para clasificación.
* **Visualización:** Gráficos interactivos de "Predicción vs. Realidad" y Feature Importance.

---

## 📸 Capturas de Pantalla

### 1. Carga y Selección de Variables
*El usuario selecciona el dataset y define qué columna quiere predecir.*
![Selección de Datos](AQUI_LINK_A_TU_SCREENSHOT_DATOS)

### 2. Configuración del Algoritmo
*Ajuste de hiperparámetros (n_estimators, max_depth) y validación cruzada (K-Fold).*
![Configuración](AQUI_LINK_A_TU_SCREENSHOT_ALGORITMO)

### 3. Resultados y Métricas
*Evaluación del desempeño del modelo con métricas estándar de la industria.*
![Resultados](AQUI_LINK_A_TU_SCREENSHOT_RESULTADOS)

---

## 🛠️ Arquitectura Técnica

El proyecto sigue una arquitectura MVC (Modelo-Vista-Controlador) adaptada a Flask:

```mermaid
graph LR
A[Cliente Web] -- HTTP POST --> B(Flask Server)
B -- Pandas --> C{Preprocesamiento}
C -- Scikit-learn --> D[Entrenamiento Modelo]
D --> E[Generación de Métricas]
E --> B
B -- HTML/JS --> A
Backend: Python con Flask y Gunicorn.

ML Core: Scikit-learn para pipelines de entrenamiento.

Data Handling: Pandas y NumPy.

Frontend: HTML5, CSS3 (Bootstrap) y Jinja2 templates.

Visualización: Matplotlib (renderizado estático) y Chart.js (dinámico).

📦 Instalación y Uso Local
Si deseas correr este proyecto en tu máquina local:

Clonar el repositorio:

Bash

git clone [https://github.com/raul-camara-20416b379/mlmc_web.git](https://github.com/raul-camara-20416b379/mlmc_web.git)
cd mlmc_web
Crear entorno virtual:

Bash

python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
Instalar dependencias:

Bash

pip install -r requirements.txt
Ejecutar servidor:

Bash

flask run
Visita http://localhost:5000 en tu navegador.

📄 Estructura del Proyecto
Plaintext

mlmc_web/
├── app.py              # Punto de entrada de la aplicación
├── core/               # Lógica de ML (Entrenamiento, Validación)
├── static/             # Archivos CSS, JS e Imágenes
├── templates/          # Plantillas HTML (Jinja2)
├── uploads/            # Carpeta temporal para datasets
└── requirements.txt    # Dependencias del proyecto
Autor: Raúl Héctor Cámara Carreón

Desarrollado como parte del portafolio de Ciencia de Datos y Desarrollo Full Stack.


### ¿Qué hace especial a este README?
1.  **Badges:** Las insignias de colores al principio (Python, Flask) le dan un look "Open Source" muy profesional.
2.  **Diagrama Mermaid:** Incluí un diagrama de flujo simple que GitHub renderiza automáticamente. Muestra que entiendes la arquitectura del sistema.
3.  **Claridad:** Explica *qué hace* y *cómo instalarlo*, que es lo que busca cualquier desarrollador que vea tu código.

¡Cópialo y dale commit! Tu repo se verá de primer nivel.

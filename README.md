# Clasificación ordinal de huevos de Aedes aegypti en ovitrampas: Un análisis comparativo mediante CNN 🦟

## 👥 Autores
* Rodrigo Carbajal
* Jorge Chamorro
* Pedro Chavez
* Nassim Ramirez
* Gonzalo Ipanaque

---

## 📝 Descripción del Proyecto
Este proyecto aborda la complejidad de la vigilancia entomológica manual del mosquito transmisor del dengue (*Aedes aegypti*). Actualmente, el proceso de conteo de huevos depositados en paletas recolectoras (ovitrampas) es lento, agotador y propenso a errores humanos, además de requerir equipos costosos como microscopios.

Para solucionar esta problemática, este repositorio contiene una solución modular basada en **Visión Computacional y Deep Learning** implementada en **PyTorch** y **TensorFlow/Keras** que automatiza el conteo de huevos mediante regresión directa y clasifica el nivel de infestación según el **Índice de Densidad de Huevos (IDH)** alineado con los criterios del Ministerio de Salud del Perú (MINSA): Bajo, Medio, Alto y Muy Alto.

---

## 🎯 Objetivos
* **Objetivo General:** Comparar el desempeño de tres arquitecturas CNN con transfer learning (*ResNet-50*, *InceptionV3* y *EfficientNet-B3*) para el conteo automatizado de huevos de *Aedes aegypti* y su posterior mapeo al riesgo epidemiológico (IDH).
* **Objetivos Específicos:**
  * Preprocesar y adaptar un dataset de imágenes de ovitrampas utilizando técnicas de aumento de datos en línea (*Online Data Augmentation*) para evitar el sobreajuste.
  * Diseñar e implementar un esquema de entrenamiento en dos fases (*Freeze & Fine-tuning*) acoplando una cabeza de regresión lineal con activación ReLU final.
  * Evaluar rigurosamente la estabilidad de los modelos mediante múltiples iteraciones estadísticas usando métricas numéricas (MAE, RMSE, MdAE) y matrices de confusión sanitarias.
  * Identificar el modelo con mayor balance entre consistencia matemática y robustez operativa para su despliegue en entornos de vigilancia real.

---

## 📂 Estructura del Repositorio
Basado en la organización del código fuente, el proyecto está estructurado de la siguiente manera:

```text
├── Dataset/                      # Dataset original en formato COCO exportado de Roboflow
│   └── _annotations.coco.json    # Archivo de anotaciones JSON
├── train/                        # Imágenes destinadas al entrenamiento (70%)
├── valid/                        # Imágenes destinadas a la validación (20%)
├── test/                         # Imágenes destinadas a la prueba final (10%)
├── prepro.py                     # Carga de datos, división aleatoria y Data Augmentation Online
├── modelos.py                    # Definición de la arquitectura unificada y control de capas congeladas
├── entrenamiento.py              # Lógica del bucle de entrenamiento (2 Fases), Huber Loss e IDH
├── main.py                       # Punto de entrada para entrenar/evaluar los 3 modelos (5 iteraciones)
├── analisis_comparativo.py       # Post-procesamiento y generación automática de gráficos estadísticos
├── estructura.txt                # Registro detallado de la jerarquía de archivos
└── resultados/                   # Almacenamiento de logs, pesos y gráficos generados
    ├── resultados_metricas.csv   # Tabla consolidada con el MAE/RMSE/MdAE de todas las corridas
    ├── predicciones_<modelo>.csv # Predicciones detalladas de la última iteración
    ├── pesos_<modelo>.pth        # Pesos guardados del mejor estado de cada arquitectura
    └── Metricas/                 # Carpeta visual con los resultados gráficos
        ├── Boxplot-ALL.png       # Boxplot comparativo global entre las 3 redes
        ├── Boxplot-<MODELO>.png  # Variabilidad interna por métrica de cada arquitectura
        └── Real-Predicho_Matriz-de-Confusion_<MODELO>.png

```

---

## ⚙️ Pipeline y Metodología Avanzada

### 1. Preprocesamiento y Augmentation Online (`prepro.py`)

En lugar de generar copias masivas de forma física offline, el pipeline realiza un **Data Augmentation Online** directo en memoria a nivel de tensor vía `torchvision.transforms` sobre las imágenes originales. Consiste en:

* Giros aleatorios (*Flips* horizontales y verticales).
* Rotación aleatoria de $\pm20^\circ$.
* Distorsión de color (*ColorJitter*) de $\pm15\%$.
* Recorte aleatorio (*Random Crop*).
* **Normalización ImageNet:** Aplicación estricta de la media y desviación estándar de ImageNet (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`), garantizando la compatibilidad con los pesos preentrenados del backend.

### 2. Arquitectura de Regresión Unificada (`modelos.py`)

Para evitar alteraciones que perjudiquen la comparación, los tres backbones comparten de forma matemática la misma ecuación y estructura para la cabeza de regresión:


$$\hat{N} = \text{ReLU}(w \cdot \text{GAP}(F_{enc}(I)) + b)$$

* **GAP:** *Global Average Pooling* para reducir las dimensiones espaciales del mapa de características.
* **Cabeza:** Capa densa (`Linear`) sin Dropout conectada a una función de activación **ReLU final**. Esto actúa como una red de seguridad explícita que garantiza que los conteos predichos sean estrictamente $\hat{N} \ge 0$.

### 3. Estrategia de Entrenamiento en 2 Fases (`entrenamiento.py`)

El modelo se entrena bajo una estrategia progresiva para maximizar la transferencia de conocimiento:

* **Fase 1 (Calentamiento - 10 Épocas):** El *encoder* se congela completamente. Solo se optimizan los parámetros de la cabeza lineal con un `Learning Rate = 1e-3` y optimizador Adam sin reducciones automáticas.
* **Fase 2 (Fine-Tuning - 50 Épocas):** Se descongela el **último tercio de los bloques del encoder** (optimizando bloques avanzados como el 6, 7 y 8 en EfficientNet-B3). Se usa un `Learning Rate = 1e-4`, un planificador `ReduceLROnPlateau` (factor=0.5, patience=3) y un mecanismo de **Early Stopping** (patience=10) enfocado en minimizar la pérdida de validación.

### 4. Función de Pérdida Adaptada: Huber Loss

A diferencia del diseño estándar con MSE (*Mean Squared Error*), se seleccionó **Huber Loss** como función de optimización. Dado que el dataset presenta un sesgo con una mediana de ~17 huevos pero valores máximos que alcanzan hasta ~500, el uso de MSE generaría gradientes desproporcionadamente enormes debido a los valores atípicos de alta densidad, empujando a los modelos a predecir únicamente la media. Huber Loss actúa de forma cuadrática para errores pequeños ($<1$ huevo) y de forma lineal para errores grandes, previniendo que los clústeres masivos desestabilicen el aprendizaje.

---

## 📊 Resultados Principales y Validación Epidemiológica

El script de control `main.py` repite de forma secuencial **5 iteraciones completas** por cada arquitectura desde cero. Al reutilizar los mismos dataloaders fijos, la estocasticidad medida responde netamente a la inicialización de la cabeza y el ordenamiento de batches, asegurando un análisis estadístico transparente.

El análisis comparativo final arrojó las siguientes conclusiones determinantes:

* ❌ **InceptionV3:** Resultó ser el modelo más inestable y errático del estudio, presentando un Error Absoluto Medio (MAE) global de **24.78** huevos y subestimando drásticamente las densidades altas.
* ⚖️ **ResNet-50:** Obtuvo el comportamiento más balanceado en el error global estándar con un MAE de **14.12** huevos, exhibiendo una alta consistencia en paletas de densidades bajas y moderadas.
* 🏆 **EfficientNet-B3 (Mejor Modelo):** Demostró la mayor precisión clínica y técnica en zonas críticas (clústeres con solapamientos severos). En escenarios de densidades extremas (por ejemplo, paletas reales con un conteo anotado de 382 huevos), EfficientNet-B3 logró predecir con éxito 339 huevos, mostrando la mejor resiliencia operativa.

### Mapeo Ordinal IDH (MINSA)

La clasificación en categorías de riesgo no se entrena en la red, sino que se mapea de forma determinística *post-hoc* sobre el conteo numérico predicho utilizando los siguientes umbrales parametrizados en el código:

* **Bajo:** $< 10$ huevos.
* **Medio:** $\ge 10$ y $< 50$ huevos.
* **Alto:** $\ge 50$ y $< 100$ huevos.
* **Muy Alto:** $\ge 100$ huevos.

**Impacto en Salud Pública:** Las evaluaciones con matrices de confusión demostraron que las desviaciones predictivas de **EfficientNet-B3** ocurren única y exclusivamente entre **categorías epidemiológicas adyacentes** (por ejemplo, clasificar una paleta de nivel 'Medio' como 'Bajo'). Se eliminaron por completo los falsos negativos extremos (como clasificar un riesgo 'Muy Alto' como 'Bajo'), garantizando que el sistema active adecuadamente los cercos sanitarios y las alertas epidemiológicas en los centros de salud sin poner en riesgo a la población.

---

## 💻 Instalación y Uso

### Requisitos Previos

* Python 3.11 o superior
* Entorno con soporte CUDA (Recomendable para aceleración por GPU, aunque cuenta con fallback automático a CPU)

### Configuración del Entorno

```bash
# 1. Clonar el repositorio
git clone [https://github.com/darkJCdark/cnn-ordinal-classifier-egg-ovitrap](https://github.com/darkJCdark/cnn-ordinal-classifier-egg-ovitrap)
cd repo-aedes-aegypti

# 2. Crear un entorno virtual de Python
python -m venv venv
source venv/bin/activate  # En Windows use: venv\Scripts\activate

# 3. Instalar dependencias requeridas
pip install opencv-python-headless pillow numpy matplotlib pandas seaborn scikit-learn torch torchvision

```

### Ejecución del Pipeline

El sistema se ejecuta en dos pasos independientes que separan el coste de cómputo del diseño gráfico:

1. **Entrenamiento y Validación Masiva:**
Ejecuta el script principal para entrenar 5 veces seguidas las arquitecturas ResNet-50, InceptionV3 y EfficientNet-B3. Al finalizar, guardará de forma automática las tablas de métricas, los pesos óptimos y los logs en la carpeta `resultados/`.
```bash
python main.py

```


2. **Generación del Análisis Estadístico y Visual:**
Una vez obtenidos los datos en la fase anterior, este script lee los archivos CSV resultantes de forma local (sin requerir PyTorch ni uso de GPU) para calcular los promedios y trazar las curvas de dispersión, los diagramas de caja comparativos y las matrices de confusión epidemiológicas.
```bash
python analisis_comparativo.py

# Colab: ML Pipeline Agent (LangChain + DeepSeek)

Agente en Colab que, dado un dataset en CSV, planifica y ejecuta un pipeline de Machine Learning de inicio a fin: análisis inicial, preprocesamiento, entrenamiento con Scikit-Learn y reporte automático con Evalcards.

<a href="https://colab.research.google.com/github/Ricardouchub/Colab-ML-Pipeline-Agent/blob/main/Colab%20ML%20Pipeline%20Agent%20Notebook.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Abrir en Google Colab"/></a>


Objetivos
-------------------------------------------------------------------------------
- Convertir un CSV en un modelo funcional sin escribir bloques de código enormes.
- Demostrar integración LLM + ML clásico: el LLM diseña el plan y Scikit-Learn lo ejecuta.
- Generar métricas y un reporte en Markdown (Evalcards) listos para documentar resultados.


Características 
-------------------------------------------------------------------------------
- Sniffing automático: detecta tipos de columnas, nulos, columnas tipo ID y candidatos a target.
- Selección interactiva de target y limpieza avanzada de columnas irrelevantes.
- Planificación con LLM (LangChain + DeepSeek): split, preprocesamiento, modelos y validación.
- Ejecución del plan con Scikit-Learn: Pipeline + ColumnTransformer + GridSearchCV.
- Reporte listo para portafolio con Evalcards (Markdown).
- Funciona para clasificación o regresión; maneja alta cardinalidad con Frequency Encoding.

Stack
-------------------------------------------------------------------------------
- **LLM Planning**: LangChain + DeepSeek (modelo deepseek-chat)
- **ML**: Scikit-Learn (Pipeline, ColumnTransformer, GridSearchCV)
- **Reports**: Evalcards (Markdown)
- **Data**: Pandas / NumPy
- **Env**: Google Colab


Flujo de trabajo
-------------------------------------------------------------------------------
1. Instalación de dependencias.
2. Configuración de la API de DeepSeek y prueba rápida del LLM.
3. Carga del CSV y vista preliminar.
4. Sniffing automático: tipos, nulos, columnas ID-like, candidatos a target y tipo de problema sugerido.
5. Selección de target y limpieza avanzada (ID-like, cardinalidad extrema, columnas sin información).
6. Planificación con LLM: se obtiene un JSON con split, preprocesamiento, modelos, CV y métricas.
7. Ejecución del plan: entrenamiento con GridSearchCV, selección de mejor modelo y evaluación; se genera reporte de Evalcards.


Entradas esperadas
-------------------------------------------------------------------------------
- Archivo CSV con columnas de atributos y, preferentemente, una columna objetivo.
- Si no hay target, el agente sugiere candidatos por heurística; el usuario elige.

Salidas
-------------------------------------------------------------------------------
- Métricas en test impresas en la notebook.
- Reporte en Markdown (Evalcards), por defecto reporte_modelo.md o en la carpeta evalcards_reports/.
- Vista previa de predicciones.


Limitaciones
-------------------------------------------------------------------------------
- Conjuntos muy grandes pueden requerir muestreo, OHE disperso o búsqueda de hiperparámetros más reducida.
- La calidad del plan depende del LLM; el notebook incluye saneo de grids para evitar fallos comunes.


Estructura del repo
-------------------------------------------------------------------------------
```
Colab-ML-Pipeline-Agent/
├── ML Pipeline Agent Notebook.ipynb     # Google Colab Notebook
├── artifacts/                           # Artefactos del test (modelo, métricas, etc.)
│   ├── model.joblib
│   ├── columns.json
│   ├── metrics_test.json
│   └── preview_predictions.csv
├── data/                                # Datos usados para el test
│   └── train.csv
├── reports/                             # Reportes Markdown del test
│   └── reporte_modelo.md
└── plan/                                # Plan generado por el LLM
    └── plan.json
```


 Q&A
-------------------------------------------------------------------------------
- ¿Sirve para clasificación y regresión?
  Sí. El tipo de problema se infiere y puede ser ajustado por el usuario.

- ¿Qué pasa si mi target es categórico con muchas clases?
  Se decide el encoding en base a cardinalidad; las de alta cardinalidad usan Frequency Encoding.

- ¿Puedo forzar modelos específicos?
  Sí. Ajuste el plan devuelto por el LLM o edite la lista de candidatos antes de entrenar.

- ¿Dónde encuentro el reporte?
  En la salida de la celda de evaluación se indica la ruta del archivo Markdown generado por Evalcards.


 Autor
-------------------------------------------------------------------------------
  **Ricardo Urdaneta**

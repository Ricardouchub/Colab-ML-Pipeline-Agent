# Colab ML Pipeline Agent (LangChain +DeepSeek)

Agente en Colab que, dado un dataset en CSV, planifica y ejecuta un pipeline de Machine Learning de inicio a fin: análisis inicial, preprocesamiento, entrenamiento con Scikit-Learn y reporte automático con Evalcards. Diseñado para portafolio: celdas cortas, explicadas y reproducibles.

-------------------------------------------------------------------------------
Objetivos
-------------------------------------------------------------------------------
- Convertir un CSV en un modelo funcional sin escribir bloques de código enormes.
- Demostrar integración LLM + ML clásico: el LLM diseña el plan y Scikit-Learn lo ejecuta.
- Generar métricas y un reporte en Markdown (Evalcards) listos para documentar resultados.

-------------------------------------------------------------------------------
Características clave
-------------------------------------------------------------------------------
- Sniffing automático: detecta tipos de columnas, nulos, columnas tipo ID y candidatos a target.
- Selección interactiva de target y limpieza avanzada de columnas irrelevantes.
- Planificación con LLM (LangChain + DeepSeek): split, preprocesamiento, modelos y validación.
- Ejecución del plan con Scikit-Learn: Pipeline + ColumnTransformer + GridSearchCV.
- Reporte listo para portafolio con Evalcards (Markdown).
- Funciona para clasificación o regresión; maneja alta cardinalidad con Frequency Encoding.
- Diseño “Colab-first”: celdas pequeñas, comentarios y salidas claras.

-------------------------------------------------------------------------------
Stack
-------------------------------------------------------------------------------
- LLM Planning: LangChain + DeepSeek (modelo deepseek-chat)
- ML: Scikit-Learn (Pipeline, ColumnTransformer, GridSearchCV)
- Reportes: Evalcards (Markdown)
- Datos: Pandas / NumPy
- Entorno: Google Colab

-------------------------------------------------------------------------------
Flujo de trabajo
-------------------------------------------------------------------------------
1. Instalación de dependencias.
2. Configuración de la API de DeepSeek y prueba rápida del LLM.
3. Carga del CSV y vista preliminar.
4. Sniffing automático: tipos, nulos, columnas ID-like, candidatos a target y tipo de problema sugerido.
5. Selección de target y limpieza avanzada (ID-like, cardinalidad extrema, columnas sin información).
6. Planificación con LLM: se obtiene un JSON con split, preprocesamiento, modelos, CV y métricas.
7. Ejecución del plan: entrenamiento con GridSearchCV, selección de mejor modelo y evaluación; se genera reporte de Evalcards.

Opcionales
- Inferencia por lotes desde CSV de test.
- Empaquetado de artefactos para descarga.

-------------------------------------------------------------------------------
Inicio rápido en Google Colab
-------------------------------------------------------------------------------
1) Abrir un notebook nuevo en Colab y ejecutar:
!pip -q install -U langchain-deepseek langchain scikit-learn evalcards python-dotenv

2) Configurar la API de DeepSeek y probar la conexión:
- Definir la variable de entorno DEEPSEEK_API_KEY en la celda correspondiente.
- Ejecutar la celda de prueba que imprime “OK”.

3) Cargar tu CSV:
- Usar la celda de carga; el lector intenta inferir el separador.
- Verificar el head() y la forma del DataFrame.

4) Sniffing automático:
- Revisar listas de columnas numéricas/categóricas, nulos y candidatos a target.
- Confirmar el tipo de problema sugerido.

5) Elegir el target y limpiar:
- Seleccionar target desde el desplegable.
- El agente elimina columnas irrelevantes:
  - Patrones de nombre (id, uuid, index, code, date, time, etc.).
  - Una sola categoría, casi todos únicos, o más del 90 % nulos.

6) Generar el plan con el LLM:
- El agente crea un JSON con:
  - split (test_size, random_state, estratificación si aplica)
  - preprocesamiento (imputación, escalado, encoding, alta cardinalidad)
  - modelos y grids
  - validación cruzada (KFold o StratifiedKFold)
  - métricas acordes al problema
  - configuración para Evalcards

7) Ejecutar el plan:
- Se construye un Pipeline reproducible y se entrena con GridSearchCV.
- Se muestra el mejor modelo y su score de CV.
- Se evalúa en test y se genera reporte_modelo.md con Evalcards.

-------------------------------------------------------------------------------
Entradas esperadas
-------------------------------------------------------------------------------
- Archivo CSV con columnas de atributos y, preferentemente, una columna objetivo.
- Si no hay target, el agente sugiere candidatos por heurística; el usuario elige.

Heurísticas y decisiones
- Columnas tipo ID se detectan por patrón de nombre y cardinalidad.
- Alta cardinalidad categórica se codifica con Frequency Encoding.
- OHE puede ser denso o disperso según dimensionalidad estimada.
- Si el plan del LLM propone hiperparámetros inválidos para un estimador, el código filtra y aplica un grid seguro por defecto.

-------------------------------------------------------------------------------
Salidas
-------------------------------------------------------------------------------
- Métricas en test impresas en la notebook.
- Reporte en Markdown (Evalcards), por defecto reporte_modelo.md o en la carpeta evalcards_reports/.
- Vista previa de predicciones.

-------------------------------------------------------------------------------
Personalización
-------------------------------------------------------------------------------
- Ajustar umbrales de alta cardinalidad y elección de encoder.
- Modificar el set de modelos del plan devuelto por el LLM.
- Cambiar número de folds, métrica primaria o grids.

-------------------------------------------------------------------------------
Limitaciones
-------------------------------------------------------------------------------
- Conjuntos muy grandes pueden requerir muestreo, OHE disperso o búsqueda de hiperparámetros más reducida.
- La calidad del plan depende del LLM; el notebook incluye saneo de grids para evitar fallos comunes.
- Si el CSV de test no incluye el target (por ejemplo, competiciones tipo Kaggle), no se calculan métricas para ese archivo.

-------------------------------------------------------------------------------
Solución de problemas
-------------------------------------------------------------------------------
- Error de columna no encontrada en ColumnTransformer:
  - Asegurarse de que el target no esté en las columnas de entrada.
  - El bloque 7A recalcula feature_cols excluyendo el target.

- Hiperparámetro inválido (por ejemplo, learning_rate en RandomForest):
  - El bloque 7A sanea los grids; re-ejecutar 6B para actualizar el plan si fuese necesario.

- Memoria insuficiente:
  - Usar OHE disperso y/o reducir grids y folds.
  - En casos extremos, activar muestreo en la celda de entrenamiento.

- Sin métricas en test.csv:
  - El archivo no tiene columna objetivo; el agente puede generar un archivo de submission, pero no métricas.

-------------------------------------------------------------------------------
Estructura del repo
-------------------------------------------------------------------------------
- notebooks/
  - colab_ml_pipeline_agent.ipynb
- data/ (opcional, no versionar datos sensibles)
- reports/ (Markdowns de Evalcards, opcional)
- README.txt (este archivo)

-------------------------------------------------------------------------------
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

# 📈 Pronóstico de Indicadores IES · UNIPUTUMAYO
**Dirección de Autoevaluación · Institución Universitaria del Putumayo**

Aplicación web (Streamlit) para que cualquier usuario cargue o escriba una serie histórica —matrícula, inscritos,
admitidos, graduados, **deserción**, retención, puntajes, variables **SNIES** u otras de IES— y obtenga un pronóstico
con el mejor modelo, el horizonte que elija y descargas listas para informes.

## ✨ Funcionalidades
- **Entrada de datos**: subir CSV/TXT/Excel, pegar desde Excel, escribir a mano en una tabla editable o usar ejemplos.
- **Reconocimiento de periodos**: `2024`, `2024-1`, `20241`, `2024S1`, `2024-I`, `2024-T3`, `2024-03`, `mar-2024`, fechas, o **AÑO + SEMESTRE** en dos columnas (formato SNIES).
- **Segmentos**: pronostica por programa, sede, facultad, etc. (o el total).
- **18 modelos** + ensamble + selección **automática** por validación retrospectiva (rolling origin):
  ingenuos, media móvil, deriva, tendencias (lineal, reciente, polinómica, exponencial), suavizado exponencial (SES, Holt, Holt amortiguado, Holt-Winters),
  Theta, ARIMA automático, Ridge autorregresivo y Bosque aleatorio sobre rezagos.
- **Métricas**: sMAPE, MAE, RMSE, MAPE, MASE · ranking y gráficos comparativos.
- **Horizonte**: el usuario escoge *hasta qué periodo* pronosticar.
- **Intervalos de confianza** (80/90/95 %) y límites lógicos según el tipo de indicador (conteo, tasa 0–100, proporción 0–1…).
- **Lectura automática** del resultado (tendencia, variación, confiabilidad).
- **Resultados**: Excel (con gráficos), CSV y gráfico HTML interactivo. **Lote** para varios indicadores × segmentos.
- Identidad institucional **UNIPUTUMAYO – Dirección de Autoevaluación**.

## 🗂️ Estructura
```
├── app.py                  # Interfaz Streamlit
├── core/
│   ├── config.py           # Nombre, dependencia, colores, límites  ← personaliza aquí
│   ├── periods.py          # Parser de periodos y series
│   ├── data_io.py          # Lectura, limpieza, agregación
│   ├── models.py           # Catálogo de modelos
│   ├── backtest.py         # Validación y ranking
│   ├── pipeline.py         # Flujo completo, intervalos, restricciones
│   ├── report.py           # Texto interpretativo y Excel
│   ├── plots.py · ui.py · catalog.py · samples.py
├── assets/logo.svg         # Logo provisional → reemplázalo por logo.png oficial
├── data/                   # Plantilla y ejemplo sintético
├── tests/                  # Pruebas (pytest)
├── .streamlit/config.toml  # Tema
└── requirements.txt
```

## ▶️ Ejecutar en local
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Desplegar en Streamlit Community Cloud
1. Crea un repositorio en GitHub y sube todo el contenido de esta carpeta:
   ```bash
   git init
   git add .
   git commit -m "App de pronóstico de indicadores IES - Uniputumayo"
   git branch -M main
   git remote add origin https://github.com/<usuario>/<repositorio>.git
   git push -u origin main
   ```
2. Entra a <https://share.streamlit.io> → **Create app** → elige el repositorio, rama `main` y archivo principal **`app.py`**.
3. En *Advanced settings* selecciona Python **3.11 o 3.12** y pulsa **Deploy**.

## 🎨 Personalización
- **Logo oficial**: guarda `assets/logo.png` (o `logo.jpg`); tiene prioridad sobre `logo.svg`.
- **Textos y colores**: `core/config.py` y `.streamlit/config.toml`.
- **Nuevos modelos**: agrega una función `f(y, h, m)` y regístrala con `_reg(...)` en `core/models.py`.

## ✅ Pruebas
```bash
pip install pytest
pytest -q
```

## ⚠️ Nota
Los conjuntos de ejemplo son **sintéticos**. Los pronósticos son estimaciones estadísticas de apoyo a la planeación y
la autoevaluación; no sustituyen el análisis experto ni constituyen cifras oficiales.

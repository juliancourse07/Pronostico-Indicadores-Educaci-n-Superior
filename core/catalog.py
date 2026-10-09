"""Catálogo orientativo de indicadores de IES (SNIES, SPADIES, Saber Pro, autoevaluación)."""

CATALOGO = [
    # (indicador, tipo sugerido, frecuencia usual, fuente típica, descripción)
    ("Inscritos", "Conteo (≥ 0, enteros)", "Semestral", "SNIES", "Aspirantes que formalizan su inscripción al programa."),
    ("Admitidos", "Conteo (≥ 0, enteros)", "Semestral", "SNIES", "Aspirantes aceptados por la institución."),
    ("Matriculados de primer curso", "Conteo (≥ 0, enteros)", "Semestral", "SNIES", "Estudiantes nuevos que inician el primer periodo del programa."),
    ("Matriculados totales", "Conteo (≥ 0, enteros)", "Semestral", "SNIES", "Estudiantes con matrícula vigente en el periodo."),
    ("Graduados", "Conteo (≥ 0, enteros)", "Anual / Semestral", "SNIES", "Estudiantes que obtienen título en el periodo."),
    ("Deserción por periodo (%)", "Tasa / porcentaje (0–100)", "Semestral / Anual", "SPADIES", "Proporción de estudiantes que abandona el programa entre periodos."),
    ("Deserción por cohorte (%)", "Tasa / porcentaje (0–100)", "Semestral", "SPADIES", "Proporción de una cohorte de ingreso que ha desertado a la fecha."),
    ("Retención primer año (%)", "Tasa / porcentaje (0–100)", "Anual", "Institucional", "Estudiantes de primer ingreso que continúan al año siguiente."),
    ("Tasa de graduación oportuna (%)", "Tasa / porcentaje (0–100)", "Anual", "Institucional", "Graduados dentro del tiempo nominal del programa sobre la cohorte."),
    ("Tasa de absorción (%)", "Tasa / porcentaje (0–100)", "Anual", "Institucional", "Admitidos o matriculados nuevos frente a inscritos."),
    ("Docentes tiempo completo equivalente", "Conteo (≥ 0, enteros)", "Semestral", "SNIES", "Planta docente en equivalente a tiempo completo."),
    ("Relación estudiante / docente", "Puntaje / índice (sin límites)", "Semestral", "Institucional", "Matriculados dividido entre docentes TCE."),
    ("Puntaje global Saber Pro", "Puntaje / índice (sin límites)", "Anual", "ICFES", "Promedio del puntaje global de los evaluados."),
    ("Ingresos / presupuesto", "Valor monetario / continuo (≥ 0)", "Anual", "Financiero", "Recursos por vigencia."),
    ("Satisfacción (%)", "Tasa / porcentaje (0–100)", "Anual", "Autoevaluación", "Percepción de estudiantes, docentes o egresados."),
]

# Evaluación con etiquetas (escena `listo`)

Herramientas que usan los cuadernos de Colab. **Cómo correr todo:** [`documentacion/ejecutar_pipeline_colab.md`](../../documentacion/ejecutar_pipeline_colab.md).

| Archivo | Qué hace |
|---|---|
| `preparar_etiquetas.py unir` | junta los `.las` (uno por pieza; la clase es el nombre del archivo) en uno solo y guarda las etiquetas |
| `preparar_etiquetas.py mapear` | pasa las etiquetas a los puntos que deja el Módulo 1 (vecino más cercano) |
| `evaluar_m4.py` | mIoU, mAcc, allAcc, ARI y NMI de cada variante y K, **y por clase: TP, FP, FN, IoU, precisión, recall, F1, FPR, FNR** (`por_clase.csv`); `--ignorar otro` deja fuera esa clase |
| `metricas.py` | las definiciones (TP/FP/FN, IoU, mAcc, allAcc) que usan los demás scripts |
| `reevaluar_desde_cruces.py` | recalcula todas las métricas desde las tablas de cruce guardadas (si faltan los grupos por punto) |
| `verificar_etapas.py` | comprueba M1 y M2 contra las etiquetas y resume M3 y M4 desde sus logs |
| `tablas_papers.py` | tablas al estilo de HyperNCD y de Sonata, FP/FN por clase y gráficos |
| `sonda_lineal.py` | sonda lineal y k-NN sobre `f_ssl` (protocolo del paper de Sonata), con división espacial; corre en Colab |
| `generar_visor_escena.py` | visor HTML de M1 a M4 con aciertos y errores (TP, FP, FN) y métricas |
| `empaquetar.py` | arma, desde el Drive, un zip con solo lo necesario de la escena (sin los `f_ssl`) |
| `traer_resultados.py` | deja ese zip (o el Drive de escritorio) en `experimentacion/datos/escena_<ESCENA>/` |

Probado en la laptop (sin GPU): unir, M1 sobre la escena, mapeo de etiquetas y el evaluador con grupos simulados.
Los resultados de la escena están en `../escena_capilla_etiquetada/` (con su propio README).

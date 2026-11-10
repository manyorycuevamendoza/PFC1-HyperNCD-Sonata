# Evaluación con etiquetas (escena `listo`)

Herramientas que usan los cuadernos de Colab. **Cómo correr todo:** [`documentacion/ejecutar_pipeline_colab.md`](../../documentacion/ejecutar_pipeline_colab.md).

| Archivo | Qué hace |
|---|---|
| `preparar_etiquetas.py unir` | junta los `.las` (uno por pieza; la clase es el nombre del archivo) en uno solo y guarda las etiquetas |
| `preparar_etiquetas.py mapear` | pasa las etiquetas a los puntos que deja el Módulo 1 (vecino más cercano) |
| `evaluar_m4.py` | mIoU, mAcc, allAcc, ARI y NMI de cada variante y K, **y por clase: TP, FP, FN, IoU, precisión, recall, F1, FPR, FNR** (`por_clase.csv`); `--ignorar otro` deja fuera esa clase |
| `metricas.py` | las definiciones (TP/FP/FN, IoU, mAcc, allAcc) que usan los demás scripts |
| `empaquetar.py` | arma, desde el Drive, un zip con solo lo necesario de la escena (sin los `f_ssl`) |
| `traer_resultados.py` | deja ese zip (o el Drive de escritorio) en `experimentacion/datos/escena_<ESCENA>/` |

Probado en la laptop (sin GPU): unir, M1 sobre la escena, mapeo de etiquetas y el evaluador con grupos simulados.
**No probado:** M2 y M3 sobre esta escena, ni los cuadernos completos en Colab.

# Evaluación con etiquetas (escena `listo`)

Herramientas que usan los cuadernos de Colab. **Cómo correr todo:** [`documentacion/ejecutar_pipeline_colab.md`](../../documentacion/ejecutar_pipeline_colab.md).

| Archivo | Qué hace |
|---|---|
| `preparar_etiquetas.py unir` | junta los `.las` (uno por pieza; la clase es el nombre del archivo) en uno solo y guarda las etiquetas |
| `preparar_etiquetas.py mapear` | pasa las etiquetas a los puntos que deja el Módulo 1 (vecino más cercano) |
| `evaluar_m4.py` | mIoU, ARI y NMI de cada variante y K; `--ignorar otro` deja fuera esa clase |

Probado en la laptop (sin GPU): unir, M1 sobre la escena, mapeo de etiquetas y el evaluador con grupos simulados.
**No probado:** M2 y M3 sobre esta escena, ni los cuadernos completos en Colab.

# Evaluación con etiquetas (escena `listo`)

Herramientas para evaluar M2, M3 y M4 con **etiquetas reales** sobre una escena etiquetada por archivos
(un `.las` por pieza; la clase es el nombre del archivo).

| Archivo | Qué hace |
|---|---|
| `preparar_etiquetas.py unir` | junta los `.las` en uno solo y guarda las etiquetas (la clase específica gana sobre `otro`) |
| `preparar_etiquetas.py mapear` | pasa las etiquetas a los puntos que deja el Módulo 1 (vecino más cercano) |
| `evaluar_m4.py` | mIoU, ARI y NMI de cada variante y K, con la referencia "todo a la clase mayoritaria" |

Cómo correr todo en Colab: [`../../documentacion/ejecutar_pipeline_colab.md`](../../documentacion/ejecutar_pipeline_colab.md).

## Probado en la laptop (sin GPU)

- Unir: 3 380 661 puntos únicos, 6 clases.
- M1 sobre la escena: 3,38 M → 1,14 M puntos en 7 s.
- Mapeo de etiquetas: distancia mediana 1,2 cm.
- Evaluador: con grupos simulados da lo esperado (perfecto por región: mIoU 0,86; al azar: 0,10).

## No probado

- M2 y M3 sobre esta escena, ni el cuaderno completo en Colab.

## Notas del dataset

- `otro.las` contiene toda la escena; las demás piezas son subconjuntos repetidos.
- No está dentro de PL7 ni de PL13 (a unos 10 m): es una escena nueva.
- 23 811 puntos aparecen en dos clases específicas a la vez.

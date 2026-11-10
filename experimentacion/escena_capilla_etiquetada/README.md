# Experimento: escena etiquetada `capilla_etiquetada` (dataset `listo`)

**Todo lo de este experimento vive en esta carpeta.** Si te pierdes, empieza aquí.

## En una frase

Probamos M1 → M2 → M3 → M4 sobre una escena de la capilla **con etiquetas reales** (arco, columna, cúpula, pared, techo y `otro`),
para medir por primera vez **qué tan bien agrupa cada módulo**, comparando Sonata **original** contra Sonata **adaptado** a la Catedral.

## Resultados clave (una sola corrida: tómalos como indicios)

Evaluación **sin la clase `otro`** (la nube completa, que puede contener elementos reales sin etiquetar), K = 8 grupos, asignación 1 a 1.
mIoU en %. La línea base "todo a una clase" da **7,9**.

| Método | Sonata original | Sonata adaptado |
|---|---|---|
| M2 (HyperNCD, solo `f_sem`) | 34,1 | 34,1 |
| M3 (solo `f_ssl`) | **53,4** | 35,6 |
| M4 base (`f_sem` + `f_geo`, la línea base de HyperNCD) | 32,1 | 32,1 |
| M4 (`f_sem` + `f_ssl`, la propuesta) | **52,5** | 36,1 |

1. **Sustituir `f_geo` por `f_ssl` ayuda mucho** con el Sonata original: M4 52,5 contra M4 base 32,1 (+20 puntos). Es el núcleo de la tesis.
2. **El Sonata adaptado empeora** (53,4 → 35,6 en M3). La adaptación no ayudó con la configuración usada (lote 1). *No está demostrado por qué.*
3. **M4 no mejora a M3 solo** (52,5 contra 53,4): **`f_sem` aporta casi nada**. Comprobado: M2 manda el **98,4 %** de los puntos a una sola cabeza y su `f_sem` tiene
   rango efectivo **3,4** (el de Sonata ronda 100). Como clasificador punto a punto, M2 queda en mIoU ≈ 9–11 %.
4. **Por clase** (sin `otro`, K = 8, 1 a 1, Sonata original en M3): cúpula 81, pared 69, techo 66, columna 32, arco 18. Las clases grandes salen bien; las **finas** (arco, columna) son las que fallan.

## Cómo leer estos números (límites)

- **Una escena, una corrida, una semilla**: sin barras de error. Diferencias pequeñas no permiten concluir.
- Los grupos se **asignan a clases con las etiquetas** (mayoría o 1 a 1); es una evaluación de agrupamiento, no de un clasificador entrenado.
- **Con regiones de 0,75 m el tope es mIoU ≈ 86–89 %** (un agrupamiento ideal por región, simulado): las regiones mezclan piezas en los bordes.
- La asignación **por mayoría** castiga a las clases chicas cuando `otro` domina (arco, columna y techo quedan en 0). Por eso se reporta también 1 a 1 y sin `otro`.
- `otro.las` contiene **toda** la escena (lo no segmentado): se ignora en la evaluación 2.
- La **sonda lineal** de Sonata (su protocolo del paper) **no se ha corrido todavía**; necesita los `f_ssl` completos en Colab.
- Los **grupos por punto** de M3 y M4 ya están (`datos/m4/.../m4_<variante>/grupos_PL13.npz`; el nombre es un resto del script, el contenido es esta escena). Se recalcularon en una segunda corrida de M4: 23 de 24 tablas de cruce coinciden exactamente con las de la primera; `adaptado`, K=6, M4 difiere en 437 de 1 120 509 puntos (no determinismo de M4).
- Los grupos son **por región de 0,75 m**, no por punto.

## Qué hay en la carpeta

```
escena_capilla_etiquetada/
├── README.md                    esta guía
├── datos/                       lo bajado de Drive (NO va a git)
│   ├── m1/ m2/ m3/ m4/          misma estructura que el Drive; m4/.../logs/ trae lo que imprimió cada módulo
│   └── MANIFIESTO.json          qué contiene el paquete y cuándo se hizo
├── resultados/                  (sí va a git: tablas y gráficos pequeños)
│   ├── evaluacion/              todas/ y sin_otro/: evaluacion.json, por_clase.csv (TP, FP, FN...), evaluacion.md, evaluacion_miou.png
│   ├── etapas/                  verificación M1 y M2 (etapas.md, etapas.json, m1_clases.png, m2_cabezas.png)
│   ├── tablas/                  estilo_hyperncd, estilo_sonata, fp_fn_por_clase (.md y .csv)
│   └── graficos/                miou_por_modulo, iou_por_clase, fp_fn_por_clase, confusion (.png)
└── visor/                       visor_escena.html (NO va a git; se regenera)
```

## Dónde mirar cada cosa

| Quiero ver... | Abre |
|---|---|
| La nube de puntos de M1 a M4, con aciertos y errores | `visor/visor_escena.html` (necesita internet para Plotly); pestaña **Lado a lado** para comparar base `f_geo`, Sonata original y adaptado |
| Si cada etapa hizo lo que debía | `resultados/etapas/etapas.md` |
| Tablas como las de los papers de HyperNCD y Sonata | `resultados/tablas/` |
| Falsos positivos y negativos por clase | `resultados/tablas/fp_fn_por_clase.md` y `resultados/graficos/fp_fn_por_clase.png` |
| Original contra adaptado | `resultados/graficos/miou_por_modulo.png` y la pestaña **Comparar** del visor |

## Cómo reproducirlo (desde la raíz del repo)

```bash
# 1. traer el paquete que baja el cuaderno 2 (celda 11) a Descargas
python experimentacion/evaluacion_etiquetada/traer_resultados.py
# 2. recalcular métricas (con FP/FN) desde las tablas de cruce guardadas
python experimentacion/evaluacion_etiquetada/reevaluar_desde_cruces.py --evaluacion experimentacion/escena_capilla_etiquetada/datos/m4/salida/capilla_etiquetada/evaluacion/evaluacion.json --salida experimentacion/escena_capilla_etiquetada/resultados/evaluacion
# 3. verificación por etapa, tablas de papers y gráficos
python experimentacion/evaluacion_etiquetada/verificar_etapas.py --datos experimentacion/escena_capilla_etiquetada/datos --salida experimentacion/escena_capilla_etiquetada/resultados/etapas
python experimentacion/evaluacion_etiquetada/tablas_papers.py --evaluacion experimentacion/escena_capilla_etiquetada/resultados/evaluacion --salida experimentacion/escena_capilla_etiquetada/resultados
# 4. visor
python experimentacion/evaluacion_etiquetada/generar_visor_escena.py --datos experimentacion/escena_capilla_etiquetada/datos --resultados experimentacion/escena_capilla_etiquetada/resultados --salida experimentacion/escena_capilla_etiquetada/visor/visor_escena.html
```

Las herramientas están en `experimentacion/evaluacion_etiquetada/` (ver su README). Cómo correr los cuadernos de Colab: `documentacion/ejecutar_pipeline_colab.md`.
Qué checkpoint y qué pesos se usaron: `documentacion/modelos_y_pesos.md` (M2 época 8; Sonata original de Meta y adaptado `sonata_catedral.pth`; verificado en los logs).

## Pendiente de este experimento

1. **Sonda lineal de Sonata** (celda 10 del cuaderno 2): el protocolo del paper, para separar "¿el descriptor sirve?" de "¿M4 lo aprovecha?".
2. **Control de la adaptación** (celda 12) y repetir la adaptación con lote mayor (ver `experimentacion/tesis2/README.md`).
3. Más escenas etiquetadas y más semillas, para poder poner barras de error.

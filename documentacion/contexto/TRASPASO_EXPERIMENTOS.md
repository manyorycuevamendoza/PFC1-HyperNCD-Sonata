# Traspaso de contexto: experimentos, variaciones y estado de la tesis

Escrito el 14-nov-2026 para que otro modelo (o persona) retome sin leer la conversación. **Todo número lleva su fuente** (archivo del repo). Lo que no se pudo verificar está marcado.

## 0. Qué cambió y por qué importa para el LaTeX

El capítulo 3 (Marco metodológico, `documentacion/tesis_latex/asesoria5/secciones/capitulo3.tex`) y su espejo, el capítulo 4, se escribieron cuando **Sonata iba congelado** y había **un solo HyperNCD**. Ya no es así:

| Antes (lo que dice el LaTeX) | Ahora (lo que se hizo) |
|---|---|
| Sonata: pesos públicos de Meta, siempre congelado | **Dos Sonata:** *original* (Meta) y *adaptado* (preentrenamiento autosupervisado continuado en la Catedral, 30 épocas) |
| Un M2: HyperNCD entrenado en SemanticPOSS, luego ArCH | **Dos HyperNCD:** *original* = SemanticPOSS (colapsa en la capilla) y *adaptado* = entrenado **desde cero** en ArCH |
| «Ninguna rama se reajusta sobre la catedral» | Falso para Sonata adaptado (se entrenó con 21 escaneos de la Catedral) |
| «Aislando una sola variable» (`f_geo` por `f_ssl`) | Diseño **2 × 2** (HyperNCD × Sonata) más un barrido de fusiones |
| Una evaluación sin etiquetas / una escena de control | Escena etiquetada `capilla_etiquetada` con 5 clases, mIoU 1 a 1, 10 semillas |

Afirmaciones **obsoletas** en `capitulo3.tex` (por línea, antes de mi borrador): «Inferencia congelada» en la figura de flujo y en E3; «Rama Auto-supervisada Congelada» (título); «Ninguna rama se reajusta sobre la catedral» (E3); fila 3 de la Tabla `tab:contrato` («Congelado»); «aislando una sola variable» (Conclusión); `\subsection[Rama Auto-supervisada Congelada]` también en `capitulo4.tex` (espejo).

## 1. Definiciones que no hay que mezclar

- **HyperNCD original** = M2 entrenado en SemanticPOSS (checkpoint `split0_run4`, época 8, mIoU de validación 0,3837). **HyperNCD adaptado** = M2 entrenado desde cero en ArCH (no se afinó el de SemanticPOSS).
- **Sonata original** = `facebook/sonata` (Hugging Face). **Sonata adaptado** = continuar el preentrenamiento con la Catedral; pesos exportados: `m3/pesos/sonata_catedral.pth` (Drive).
- **M2** = solo `f_sem` (96 dim.); **M3** = solo `f_ssl` (1088 dim.); **M4 base** = `f_sem` + `f_geo`; **M4** = `f_sem` + `f_ssl`.
- **Evaluación:** capilla etiquetada, K = 8 grupos, emparejamiento húngaro (1 a 1), **sin la clase `otro`**, regiones de 0,75 m (≥ 30 puntos). mIoU = promedio del IoU de 5 clases: pared, techo, cúpula, arco, columna.
- **Semillas:** son de la inicialización del agrupamiento (Sinkhorn), **no** reentrenamientos de HyperNCD ni de Sonata.

## 2. Inventario de experimentos (en orden)

| Id | Qué | Resultado clave | Dónde |
|---|---|---|---|
| X0 | Preprocesamiento M1 de la Catedral (23 LAS, SOR, voxel 5 cm, normales) | matriz `S` de 9 columnas; PL7 y PL13 como escenas de trabajo | `preprocesar.py`; `experimentacion/datos/` |
| X1 | **Diagnóstico de dominio:** M2 de SemanticPOSS en la capilla | 98,4 % de los puntos en una cabeza; NMI 0,018; en SemanticPOSS 30,4 % y NMI 0,48 | `experimentacion/m2_dominio/README.md` |
| X2 | **Entrenar HyperNCD en ArCH** (Colab T4) | 30 épocas, 75 min, RAM 4,9 GB, GPU 9,5 GB; mIoU de validación 0,489 (mejor época 20; última 0,485) | `experimentacion/arch_entrenamiento/README.md` |
| X3 | **M2 ArCH en la capilla** | cabeza dominante 44,6 %; NMI 0,191 (0,439 sin `otro`); 12,6 % de puntos en cabezas novedosas | `arch_entrenamiento/resultados/m2_capilla/` |
| X4 | **Adaptar Sonata** a la Catedral | 21 escaneos (todos menos PL7 y PL13), 30 épocas × 105 it., **lote 1**, LR 2e-4, 32 768 puntos/vista, bf16, atención SDPA, GPU L4; 103 min; pesos cambian 8,35 %; coseno con el original 0,54; rango efectivo 110,5 → 81,3 | `experimentacion/adaptacion_sonata/README.md` |
| X5 | **M4 en la capilla etiquetada**, 2 × 2 | ver §3 | `arch_entrenamiento/resultados/m4_capilla/`, `escena_capilla_etiquetada/` |
| X6 | **Barrido de fusiones deterministas** (39 configuraciones × 4 combinaciones × 5 semillas, dos mitades espaciales) | ninguna mejora significativa con el M2 sano | `experimentacion/modulo4_fusion/README.md` |
| X7 | **Referencias con 10 semillas** (M2, M3, M4 base, M4) | corrige las cifras de una semilla | `modulo4_fusion/resultados/referencias_semillas.md` |
| X8 | **Caso Gráfico:** umbral de 70 % de IoU por clase, dificultad por clase, geometría de las clases | ver `documentacion/caso_grafico_renzo.html` | commit `4114abf` |

Registro de todo: `experimentacion/registro/INDICE.md` (20 experimentos) y visor `experimentacion/visor/index.html`. Dónde está lo pesado (Drive / local / git): `documentacion/drive_indice.md`.

## 3. Resultados que el LaTeX debe usar (K = 8, 1 a 1, sin `otro`, media ± desviación de 10 semillas)

Fuente: `experimentacion/modulo4_fusion/resultados/referencias_semillas.md`.

| Combinación | M2 | M3 | M4 base | M4 |
|---|---|---|---|---|
| HyperNCD original + Sonata original | 35,9 ± 0,6 | 45,4 ± 8,5 | 32,5 ± 0,6 | 43,4 ± 8,2 |
| HyperNCD original + Sonata adaptado | 35,9 ± 0,6 | 34,3 ± 1,7 | 32,5 ± 0,6 | 33,7 ± 2,1 |
| HyperNCD **adaptado** + Sonata original | 48,4 ± 1,1 | 45,4 ± 8,5 | 51,4 ± 1,7 | **52,5 ± 2,5** |
| HyperNCD **adaptado** + Sonata adaptado | 48,4 ± 1,1 | 34,3 ± 1,7 | 51,4 ± 1,7 | 47,0 ± 0,8 |

Conclusiones sostenidas por los datos (una escena; desviaciones solo de inicialización):
1. Un M2 que no colapsa cambia el resultado: M2 solo sube de 35,9 a 48,4.
2. Con el M2 sano M4 (52,5) supera a M3 (45,4) y a M2 (48,4) y es mucho más estable que M3 (± 2,5 contra ± 8,5), pero **no supera de forma significativa** a M4 base (+1,1, dentro del ruido).
3. Con el M2 colapsado, `f_ssl` sí ayuda a M4 frente a la línea base (43,4 contra 32,5).
4. **Sonata adaptado es peor** que el original (M3: 34,3 contra 45,4; M4 con M2 sano: 47,0 contra 52,5). Causa **no demostrada** (lote 1; falta el control de entorno).
5. Barrido de fusiones: ningún método (centrado, PCA con blanqueo, producto, mínimo, confianza de M2, SNF) mejora de forma significativa con el M2 sano; con el M2 colapsado «rescatan» a M4 de ~42 a ~50.
6. **Las cifras de una semilla que circularon (M3 original 53,4; M4 con SemanticPOSS 52,5) eran semillas afortunadas.** No usarlas.

## 4. Datos de las escenas

- **Capilla etiquetada («listo»):** entregada por el **asesor Fiestas**, etiquetada a mano; el paquete que se recibió es `listo.zip` (carpeta `listo/` con 19 `.las`: arco1–7, arco89, columna1–2, cupula, pared1–6, techo, más `otro.las` con la escena completa de 3 380 661 puntos). Tras M1: 1 135 946 puntos con etiqueta. **Falta el nombre del archivo original.**
- **ArCH** (Politecnico di Torino, CC BY-NC-SA 4.0; `matrone2020benchmark` ya está en `referencias.bib`): 17 escenas (15 de entrenamiento y 2 de prueba), 10 clases (se ignora `other`), bloques de 10 m, voxel 5 cm, 163 bloques de entrenamiento y 29 de validación. Novedosas: arco, columna y bóveda.
- **Tesis de patrimonio (UTEC):** Alvarado Ramos y Cuaresma Puclla (2024), asesor Fiestas Iquira; **sin link**, copia local en `documentacion/papers/`.

## 5. Decisiones y reglas de trabajo del usuario (cumplirlas)

- Responder en **español simple y directo**; el usuario escribe con muchos errores de tipeo.
- **Nunca** añadir «Co-Authored-By: Claude» ni atribución en commits (aparece en Contributors de GitHub). **No hacer commit ni push sin que lo pida.** No reescribir historial.
- **Estructura de la tesis:** experimentación y validación van **repartidas** en el capítulo de metodología (3) y en el de resultados (4), dentro de las secciones existentes; no crear secciones nuevas para ellas.
- Reportar resultados negativos tal cual (la adaptación de Sonata empeora; la propuesta no supera a su base).
- Todo experimento nuevo: carpeta propia con README, registro en `experimentacion/registro/` (`registrar.py`) y ruta de sus datos pesados.
- Celdas de Colab completas; credenciales nunca en el chat ni en el código.
- Las medias siempre con desviación y número de semillas; decir cuando algo es de una sola semilla.

## 6. Pendientes

1. **Control de la adaptación de Sonata:** extraer PL13 con pesos originales en el mismo entorno de Colab (separa adaptación de entorno). `experimentacion/tesis2/README.md` §1.
2. Repetir la adaptación con **lote mayor** (Khipu, `sonata_adaptacion/`) y con otras semillas; sonda lineal de Sonata.
3. Más escenas etiquetadas y más semillas; hoy todo sale de una escena.
4. Fusión **aprendida** (tesis 2): entrenar HyperNCD con `[f_sem; f_ssl]` como entrada; requiere `f_ssl` sobre ArCH.
5. Nombre del archivo original de la capilla; verificar las referencias externas del Caso Gráfico (salen de búsquedas); revisar a ojo las figuras de los PDF (no hay OCR).
6. **LaTeX:** alinear capítulo 4, resumen, abstract, conclusiones, trabajos futuros y anexos con el diseño 2 × 2 (ver §7).

## 7. Estado del LaTeX tras el borrador de esta fecha

**Regla de estructura (del usuario):** la experimentación y la validación **no son secciones nuevas**: se reparten en las secciones que ya existen, el método en el capítulo 3 y los resultados en el capítulo 4, con los mismos títulos (`\espejo`). Una primera versión del borrador las puso en una sección aparte y se **deshizo**; no volver a crearla.

**Dónde quedó cada cosa (capítulos 3 y 4, mismos títulos en ambos):**

| Contenido | Capítulo 3 (método) | Capítulo 4 (resultados) |
|---|---|---|
| Factores (HyperNCD × Sonata × fusión) y tabla de experimentos X0 a X8 | `Procedimiento Experimental`, párrafo «Factores y variantes» (Tablas `tab:factores` y `tab:experimentos`) | lo cubren las secciones de abajo |
| HyperNCD original y adaptado | `Rama Semántica`, párrafo final | `Rama Semántica` (ya existía) y la lista de E1 a E3 |
| Sonata original y adaptado; procedimiento y resultados de la adaptación (X4) | `Rama Auto-supervisada: Sonata Original y Adaptado`, subsubsección «Adaptación de Sonata a la Catedral» (`sec:variante-sonata`) | misma subsección: Tabla `tab:sonata-adaptado` |
| Barrido de fusiones (X6) | `Punto de Fusión`, subsubsección «Barrido de fusiones» (`sec:barrido-fusion`) | misma subsección: Tabla `tab:barrido`, Figura `fig:barrido-combos` |
| Escena etiquetada y métricas | `Trazabilidad de la Métrica`, «Escena etiquetada y métricas» (`sec:escena-etiquetada`) | misma subsección, párrafo corto |
| Umbral de 70 % y dificultad por clase (X8) | `Trazabilidad de la Métrica`, «Umbral y dificultad por clase» (`sec:umbral-dificultad`) | misma subsección: Tablas `tab:umbral` y `tab:geometria`, Figuras `fig:caso-iou` y `fig:caso-geometria` |

Otros cambios del borrador en `capitulo3.tex`: introducción del capítulo; figura de flujo (E2 y E3); fila 3 de `tab:contrato`; Paso 2 de la rama auto-supervisada; lista E2/E3 y «Protocolo de Repetición»; riesgo (iv) (adaptación con lote pequeño); conclusión (ya no «una sola variable»). En `capitulo4.tex`: título de «Rama Auto-supervisada: Sonata Original y Adaptado». Imágenes nuevas en `asesoria5/imagenes/`: `fusion_barrido_combos.png`, `caso_grafico_iou_10semillas.png`, `caso_grafico_geometria.png`. Bibliografía: 3 entradas nuevas en `referencias.bib` (`wang2014snf`, `kumar2011coregularized`, `chao2017survey`) con `% TODO verificar` (autores, título, volumen y páginas están en parte de memoria). El producto, el mínimo y el peso por confianza del M2 son variantes **propias**: no citar un artículo para ellas. No se añadieron DSC-Net, Betsas ni GrowSP/PointDC/LogoSP porque el texto no los usa.

**El capítulo 4 ya traía** la tabla 2 × 2 con diez semillas (`tab:m2-m4-semillas`), E1 a E3 y el IoU por clase; sus cifras coinciden con §3 de este documento.

**Dónde queda la palabra «congelado» y hay que reencuadrarla** (Sonata ya tiene una variante adaptada; «congelado» solo es cierto *durante la extracción y la fusión*): `introduccion.tex` líneas 54, 85, 99, 106, 107, 122; `capitulo1.tex` líneas 33, 129, 160–176, 202, 294, 334–339 (incluye «régimen congelado, que es el que adopta este proyecto»); `capitulo2.tex` líneas 178, 189–190 (`sec:congelado`), 423, 497–504, 582–594; `capitulo4.tex` líneas 21, 68, 81–82 («Sonata congelado, original y adaptado»), 212; `conclusiones.tex` línea 18; `anexos.tex` línea 170 (nomenclatura de $f_{\mathrm{ssl}}$); `anexo_pasos.tex` línea 113. Es una decisión de fondo: o se redefine «congelado» como «sin actualizarse durante la extracción», o se reescribe el régimen del proyecto como «original congelado y adaptado».

Lo que debe hacer el revisor, en orden:
1. Compilar. **Resultado del borrador (14-nov-2026): compila sin errores propios, 127 páginas, 0 referencias y 0 citas sin resolver; el único aviso es uno anterior, `secciones/anexos.tex:231` («Infinite glue shrinkage»).** La sección nueva queda en 3.7.2 (páginas impresas 51 a 55). En este equipo `latexmk` **no funciona** (falta Perl); usar, desde `documentacion/tesis_latex/asesoria5/`: `pdflatex main.tex`, `bibtex main`, `pdflatex main.tex`, `pdflatex main.tex`. No usar `-output-directory` (deja referencias sin resolver). Los auxiliares y `main.pdf` están en `.gitignore`.
2. Decidir y aplicar el reencuadre de «congelado» en los archivos listados arriba.
3. Revisar contra sus fuentes los resultados de X4, X6 y X8 ya repartidos en el capítulo 4 (tabla de la sección anterior) y comprobar que el resto del capítulo 4 no los contradiga (por ejemplo, las líneas con «Sonata congelado»).
4. Reescribir `resumen.tex`, `abstract.tex`, `conclusiones.tex`, `trabajosFuturos.tex` y la `introduccion.tex` (objetivos, hipótesis) para el diseño 2 × 2 y el resultado «M4 supera a M3 y M2 pero no a su base».
5. Revisar anexos (`anexo_parametros.tex`, `anexo_complementarios.tex`): parámetros de la adaptación de Sonata y del barrido de fusiones.
6. Comprobar cada número contra su archivo fuente.

## 8. Prompt para pegar al modelo revisor

> Lee `documentacion/contexto/TRASPASO_EXPERIMENTOS.md` completo y luego `documentacion/tesis_latex/asesoria5/secciones/capitulo3.tex` y `capitulo4.tex`. Eres el revisor de un primer borrador: corrige el Marco metodológico de la tesis para que documente **todos** los experimentos X0 a X8 y sus variaciones (HyperNCD original/adaptado, Sonata original/adaptado, fusiones, semillas), sin decir que Sonata está congelado. El capítulo 3 no reporta valores medidos; los resultados van en el capítulo 4, con el mismo título por sección (`\espejo`). Después alinea el capítulo 4, el resumen, el abstract, la introducción, las conclusiones y los trabajos futuros. Verifica cada número contra su archivo fuente (§2 y §3 del traspaso). Reglas: la experimentación y la validación **no son secciones nuevas**: van repartidas en las secciones existentes de los capítulos 3 y 4. Español simple; no agregues Co-Authored-By ni atribución a Claude en commits; no hagas commit ni push sin que te lo pidan; informa resultados negativos tal cual; no inventes citas (marca `% TODO verificar cita`). Compila desde `documentacion/tesis_latex/asesoria5/` con `pdflatex`, `bibtex`, `pdflatex`, `pdflatex` (`latexmk` no funciona: falta Perl) y arregla los errores. Puedes usar el agente `profesor-evaluador` al final para evaluar coherencia entre capítulos, resultados y código.

## 9. Reescritura del Marco metodológico (14-nov-2026, correcciones del usuario)

`capitulo3.tex` se reescribió completo con estas reglas, que valen para el resto de la tesis.

- **Sin «:» ni «;» en el texto.** Solo punto y coma simple. Lenguaje natural, párrafos cortos, sin relleno.
- **Flujo general (Figura `fig:flujo`)**: E0 datos, E1 preparación (M1), E2 modelo semántico (M2), E3 modelo auto-supervisado (M3), E4 fusión y agrupamiento (M4), E5 evaluación. **El diagnóstico de dominio ya no es una etapa**, es un hallazgo y va en resultados. El capítulo 4 se ajustó a esta numeración en su sección de flujo.
- **Orden fijo M1, M2, M3, M4** en todo (tabla de parámetros, replicación, experimentos).
- **Predominan flujos, tablas y pseudocódigo**. Nuevos en el capítulo 3: Figura `fig:dependencias` (M2 y M3 en paralelo, ambos dependen de M1), Tabla `tab:datos-antes-despues`, Figura `fig:flujo-m2`, Tabla `tab:datasets-m2`, Algoritmos `alg:inferencia-sonata`, `alg:revision`, `alg:fusion`, `alg:prueba`, `alg:experimento`, `alg:evaluacion`, Tabla `tab:observables` (antes «Observables», ahora «Mediciones de control de cada extracción»), Figuras `fig:flujo-adaptacion` y `fig:flujo-prueba`.
- **Se quitaron del capítulo 3** las secciones «Riesgos y Mitigación» y «Conclusión del Marco Metodológico», y el párrafo que explicaba el espejo con el capítulo 4. En el capítulo 4 esas dos secciones siguen, sin enlace de espejo. Falta decidir con el usuario si también se quitan de resultados.
- **Escena de prueba**: es la capilla etiquetada («listo»). PL7 y PL13 solo sirvieron para revisar descriptores sin etiquetas.
- **Pendiente de verificar**: las 4 clases novedosas de la partición 0 de SemanticPOSS (hay un `% TODO` en la tabla), y el título «Rama Auto-supervisada, Sonata Original y Adaptado», que cambió en los dos capítulos.
- Comprobación de dos puntos con un script (0 en el texto del capítulo 3). El capítulo 4 y los demás **no** se han limpiado de «:» ni «;».
- Compila: 132 páginas, 0 referencias sin resolver, 0 etiquetas duplicadas.
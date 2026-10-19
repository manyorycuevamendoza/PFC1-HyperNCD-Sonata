# Ver y explicar los resultados (Módulo 1 y Módulo 3)

> **Para el asistente de IA que lea esto:** el usuario quiere, **primero, VER**
> la vista 3D interactiva, y **después** que se la expliques. Sigue los pasos en
> orden. No arranques servidores web: la página es un `.html` que se abre con
> doble clic. Contexto completo y trampas conocidas en [README.md](README.md).

---

## Paso 1: Mostrar la vista (lo primero)

### 1.a. ¿Ya existe?

Busca `experimentacion/resultados/visor_PL13.html` (todas las etapas de PL13) o
`visor_PL7.html` (solo Módulo 1). Si existe, ve al paso 1.c. Los `.html` **no
están en git** (pesan ~10 MB), así que en un clon nuevo hay que generarlos.

Si solo se necesitan imágenes fijas, ya están en git:
`experimentacion/resultados/informe/lamina_*.png`.

### 1.b. Generarla

Hacen falta datos que **no están en git** y viven en el clúster Khipu de UTEC,
en `/home/renzo.felix/PFC1-HyperNCD-Sonata/`. Pídele al usuario que los baje con
**su propia cuenta**, en una terminal **de su PC** (no dentro de Khipu), desde
la raíz del repo. Cambia `USUARIO` por su usuario; cada comando pide su
contraseña (que la escriba él; nunca se la pidas):

```
mkdir experimentacion\datos\modulo3_PL13
scp USUARIO@khipu.utec.edu.pe:/home/renzo.felix/PFC1-HyperNCD-Sonata/cache/PL13_feats.npz experimentacion/datos/modulo3_PL13
scp USUARIO@khipu.utec.edu.pe:/home/renzo.felix/PFC1-HyperNCD-Sonata/data/PL13_SCN0001.npy experimentacion/datos/modulo3_PL13
scp USUARIO@khipu.utec.edu.pe:/home/renzo.felix/PFC1-HyperNCD-Sonata/data/PL13_SCN0001.json experimentacion/datos/modulo3_PL13
scp "USUARIO@khipu.utec.edu.pe:/home/renzo.felix/PFC1-HyperNCD-Sonata/datos_crudos/Post Procesado/Post Procesado GLS-2000/PL13*.las" experimentacion/datos/modulo3_PL13
```

- El último usa `*` porque el archivo se llama `PL13'_SCN0001.las` (con
  apóstrofo) y el `scp` de Windows no lo escapa bien. Renómbralo a
  `PL13_SCN0001.las`. Es opcional: sin él no sale la etapa "Crudo".
- Si sale "Permission denied" al leer la carpeta de `renzo.felix`, Renzo tiene
  que darle acceso o pasarle los archivos por otro medio.

> ⚠️ **Usa el `.npy` de Khipu (`data/`), no uno regenerado en otra máquina.**
> Los descriptores van amarrados al orden de filas de ese archivo; otra versión
> de Open3D da los mismos puntos en otro orden y el Módulo 3 sale como ruido.
> El visor lo detecta y se detiene con un aviso.

Instala y genera (desde la raíz del repo):

```
pip install numpy laspy plotly
python experimentacion/visor_3d.py --las experimentacion/datos/modulo3_PL13/PL13_SCN0001.las --npy experimentacion/datos/modulo3_PL13/PL13_SCN0001.npy --json experimentacion/datos/modulo3_PL13/PL13_SCN0001.json --feats experimentacion/datos/modulo3_PL13/PL13_feats.npz --output experimentacion/resultados/visor_PL13.html
```

Tarda uno o dos minutos y usa unos 8–10 GB de RAM. Imprime `[ok] ...html`.
Si no se bajó el `.las`, quita `--las ...`.

### 1.c. Abrirla

**Doble clic** en `experimentacion/resultados/visor_PL13.html` (necesita
internet: carga la librería Plotly).

- **Girar:** clic izquierdo y arrastrar. **Zoom:** rueda. **Desplazar:** clic derecho.
- **Botones de arriba:** cambian la etapa. **El ángulo se mantiene** al cambiar.
- **📷 arriba a la derecha:** baja la vista actual en PNG.

---

## Paso 2: Explicar lo que se ve

### Contexto en una frase

Es una **iglesia colonial (Catedral de Lima)** escaneada con láser. El escaneo
pasa por el **Módulo 1** (limpieza y reducción) y luego por **Sonata**
(Módulo 3), que le asigna a **cada punto 1088 números** que describen "qué es".

### Los botones, en orden

| Botón | Etapa | Cómo leerlo |
|---|---|---|
| **Crudo** | Entrada (`.las`) | Lo que midió el escáner. Muchos puntos cerca del escáner, pocos lejos |
| **M1: color real** | Salida del Módulo 1 | Limpia (sin puntos sueltos) y con densidad pareja: un punto por cubo de 5 cm |
| **M1: normales** | Salida del Módulo 1 | Color = hacia dónde mira la superficie. Pisos de un color, paredes de otro |
| **M3: PCA de f_ssl** | Salida de Sonata | Los 1088 números comprimidos a 3 y usados como color. **Mismo color = Sonata cree que son parecidos** |
| **M3: similitud con ◆** | Salida de Sonata | Rojo = parecido al rombo negro, azul = distinto. Es el experimento del "sofá" del paper de Sonata: si se encienden otros arcos y no todo lo que está a la misma altura, el modelo no hace trampa con la geometría |
| **M3: k-means** | Salida de Sonata | 8 grupos formados **sin etiquetas**. Si coinciden con arcos, piso y muros, Sonata ya separa partes de la iglesia sola |

### Qué es Sonata (por si pregunta)

- Sonata (Wu et al., CVPR 2025) **no inventó una arquitectura**: usa la red
  PTv3. Lo nuevo es **cómo la entrenaron**: sin etiquetas, solo el encoder, y
  ocultando pistas geométricas fáciles para que aprenda qué son las cosas.
- Aquí se usan sus **pesos ya entrenados**, congelados. No se entrenó nada.

### Qué se observó en PL13

- **PCA:** nervaduras de distintas bóvedas en verde y azul, piso en amarillo,
  muros en rojo.
- **Similitud:** el ◆ cae en una nervadura; se parecen otras nervaduras, no el piso.
- **k-means:** separa bóvedas, piso y muros en grupos distintos.
- **Medidas** (`resultados/modulo3_PL13.json`): rango efectivo 110,6 de 1088,
  las features **no colapsaron** (un modelo colapsado daría ~1).

### Qué NO se puede decir todavía

- Es una revisión **visual, cualitativa**. No hay "% de acierto" porque los
  datos **no tienen etiquetas**. Para eso hace falta *linear probing*.
- Cada etapa dibuja 100 000 puntos al azar, no todos.

### Dónde encaja en la tesis

**Módulo 1** → **Módulo 3** (Sonata) → **Módulo 4** (HyperNCD: regiones,
prototipos e hipergrafo para descubrir clases nuevas). Las mismas imágenes
están en `LATEX/asesoria5/secciones/capitulo3.tex` (secciones "Visor 3D del
Proyecto", "Entrada y Salida de la Prueba" y "Rama Auto-supervisada Congelada").

# Ver y explicar los resultados de Sonata (Módulo 3)

> **Para el asistente de IA que lea esto:** el usuario quiere, **primero, VER**
> la visualización 3D interactiva de las features de Sonata, y **después** que
> se la expliques. Sigue los pasos en orden. No arranques servidores web: la
> página es un archivo `.html` que se abre directamente en el navegador.

---

## Paso 1: Mostrar la visualización (lo primero)

### 1.a. ¿Ya existe el `.html`?

Busca `experimentacion/resultados/modulo3_PL13.html`. Si existe, salta al
paso 1.c. El `.html` **no está en git** (pesa 11 MB), así que en un clon nuevo
normalmente no estará y hay que generarlo (paso 1.b).

### 1.b. Generarlo

Hacen falta dos archivos de datos que **no están en git** (son pesados) y viven
en el clúster Khipu de UTEC, en `~/PFC1-HyperNCD-Sonata/`:

| Archivo en Khipu | Qué es | Tamaño |
|---|---|---|
| `cache/PL13_feats.npz` | Salida de Sonata: 1088 números por punto | ~680 MB |
| `data/PL13_SCN0001.npy` | Nube preprocesada (para el color) | ~29 MB |

Pídele al usuario que los baje con **su propia cuenta** de Khipu. Los comandos
se corren en una terminal **de su PC** (PowerShell o cmd), **no dentro de
Khipu**, y desde la raíz del repo. Cambia `USUARIO` por su usuario de Khipu;
cada comando pedirá su contraseña (que la escriba él, nunca se la pidas):

```
mkdir experimentacion\datos\modulo3_PL13
scp USUARIO@khipu.utec.edu.pe:/home/renzo.felix/PFC1-HyperNCD-Sonata/cache/PL13_feats.npz experimentacion/datos/modulo3_PL13
scp USUARIO@khipu.utec.edu.pe:/home/renzo.felix/PFC1-HyperNCD-Sonata/data/PL13_SCN0001.npy experimentacion/datos/modulo3_PL13
```

> Si su cuenta no puede leer la carpeta de `renzo.felix` (error "Permission
> denied"), Renzo tiene que darle acceso o pasarle los dos archivos por otro
> medio (Drive, USB).

Instala lo necesario y genera la vista (desde la raíz del repo):

```
pip install numpy matplotlib plotly
python experimentacion/modulo3_sonata/visualizar_features.py --feats experimentacion/datos/modulo3_PL13/PL13_feats.npz --data experimentacion/datos/modulo3_PL13/PL13_SCN0001.npy --output experimentacion/resultados/modulo3_PL13 --html
```

Tarda uno o dos minutos y usa unos 8 a 10 GB de RAM. Al terminar imprime
`[ok] ...modulo3_PL13.html`.

### 1.c. Abrirlo

Dile al usuario que haga **doble clic** en
`experimentacion/resultados/modulo3_PL13.html` (o que lo arrastre al
navegador). Necesita internet: la página carga la librería Plotly.

Cómo se usa:
- **Girar:** clic izquierdo y arrastrar.
- **Zoom:** rueda del mouse.
- **Desplazar:** clic derecho y arrastrar.
- **Botones de arriba:** cambian cómo se pinta la nube (ver paso 2).

> No uses `localhost` ni `python -m http.server`. No hace falta, y un servidor
> temporal deja de funcionar en cuanto se cierra.

Hay también una imagen fija con las 4 vistas: `experimentacion/resultados/modulo3_PL13.png`
(esta sí está en git).

---

## Paso 2: Explicar lo que se ve

### Contexto en una frase

Es una **iglesia colonial (Catedral de Lima)** escaneada con láser. Se pasó
por **Sonata**, un modelo que mira la nube de puntos y le asigna a **cada
punto 1088 números** que describen "qué es" ese punto. Nadie puede ver 1088
números, así que los botones muestran 4 formas de resumirlos.

### Qué es Sonata (por si pregunta)

- Sonata (Wu et al., CVPR 2025) **no inventó una arquitectura nueva**: usa la
  red PTv3. Lo nuevo es **cómo la entrenaron**: sin etiquetas
  (*self-supervised*), solo con el encoder, y ocultando pistas geométricas
  fáciles para que aprenda **qué son las cosas** y no solo su forma o altura.
- Aquí se usan sus **pesos ya entrenados**, congelados. No se entrenó nada.

### Los 4 botones

| Botón | Qué muestra | Cómo leerlo |
|---|---|---|
| **Color real** | El color que midió el escáner | Referencia: para saber qué es cada cosa (arco, piso, muro) |
| **PCA de f_ssl** | Los 1088 números comprimidos a 3 y usados como color | **Mismo color = Sonata cree que esos puntos son parecidos.** Si arcos de bóvedas distintas salen del mismo color, Sonata "entiende" que son lo mismo |
| **Similitud con ◆** | Qué tan parecido es cada punto al punto negro ◆ (similitud coseno: +1 igual, 0 sin relación, −1 opuesto) | **Rojo = parecido al ◆, azul = distinto.** Es el mismo experimento del "sofá" del paper de Sonata: si se encienden otros arcos y no todo lo que está a la misma altura, el modelo no hace trampa con la geometría |
| **k-means (k=8)** | Los puntos repartidos en 8 grupos según sus 1088 números, **sin ninguna etiqueta** | Si un grupo coincide con "los arcos" y otro con "el piso", Sonata ya separa partes de la iglesia por sí sola |

### Qué se observó en PL13

- **PCA:** los arcos de distintas bóvedas salen rosados, el piso morado y los
  muros verdes o celestes.
- **Similitud:** el ◆ cae en una nervadura; se parecen otros arcos, no el piso.
- **k-means:** separa arcos, piso y muros en grupos distintos.
- **Medidas** (`experimentacion/resultados/modulo3_PL13.json`): rango efectivo
  110,6 de 1088, es decir, las features usan unas 110 direcciones distintas y
  **no colapsaron** (un modelo colapsado daría ~1).

### Qué NO se puede decir todavía

- Es una revisión **visual, cualitativa**. No hay un "% de acierto" porque los
  datos de la iglesia **no tienen etiquetas**. Para medir acierto hace falta
  *linear probing* con puntos etiquetados.
- El ◆ se eligió al azar. Se puede fijar con `--consulta N`.

### Dónde encaja en la tesis

Pipeline: **Módulo 1** (limpia la nube) → **Módulo 3** (Sonata, esto) →
**Módulo 4** (HyperNCD agrupa en regiones, prototipos e hipergrafo para
descubrir clases nuevas). El k-means de aquí es un agrupamiento "a ciegas";
HyperNCD debería agrupar mejor porque suma conocimiento de clases
etiquetadas y relaciones entre regiones. Más detalle en
`LATEX/asesoria5/secciones/capitulo3.tex` (sección "Rama Auto-supervisada
Congelada").

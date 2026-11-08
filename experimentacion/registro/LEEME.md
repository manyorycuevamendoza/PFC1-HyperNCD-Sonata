# Registro de experimentos

Un archivo `<id>.json` por experimento. El visor (`../visor/index.html`) lee de aquí; no hay
base de datos. **Si no está registrado aquí, no existe para el visor ni para el equipo.**

## Regla de proyecto

Cada experimento nuevo deja siempre tres cosas:

1. **Su registro** en esta carpeta (`<id>.json`, campos abajo).
2. **Su documento** (README) con la pregunta, el resultado y los límites. El registro lo enlaza en
   `resultados.documento`.
3. **Sus datos pesados guardados en un lugar fijo** (Drive o Khipu) con la ruta anotada en
   `datos_pesados`. Los `.npz`, `.npy`, `.pth` y `.html` no van a git, así que sin esa ruta se pierden.

Luego se corre `python experimentacion/registro/registrar.py`, que valida y regenera el índice.
Un experimento **no se edita para "mejorarlo"**: si cambias algo, es un experimento nuevo (otro `id`).

## Campos del registro

| Campo | Qué es |
|---|---|
| `id` | igual al nombre del archivo; `m<etapa>_<qué>` (p. ej. `m3_adaptado_e30`) |
| `etapa` | `M1` a `M5` |
| `titulo`, `descripcion` | qué se hizo, en una línea y en un párrafo corto |
| `estado` | `hecho`, `en curso`, `descartado`… |
| `fecha` | `AAAA-MM-DD` |
| `entradas` | `id` de los experimentos de los que viene. Así M4 dice qué M2 y qué M3 usó |
| `parametros` | los valores que definen la corrida |
| `hardware`, `costo` | dónde corrió y cuánto costó (`null` = aún sin medir; no se inventa) |
| `metricas` | números clave para comparar (se muestran lado a lado en el visor) |
| `resultados` | `documento`, `json`, `imagenes`, `visores` (rutas desde la raíz del repo) |
| `datos_pesados` | `que` y `donde` (Drive, Khipu, local) |
| `notas` | límites y advertencias |

## Cómo agregar uno

1. Copia un registro parecido (`m3_adaptado_e30.json` es el más completo) y cambia los campos.
2. Pon `entradas` con los `id` correctos.
3. Corre `registrar.py`. Si falla, dice qué campo falta o qué entrada no existe.

## Archivos generados (no editar)

`indice.json`, `indice.js` (lo usa el visor al abrirlo con doble clic) e `INDICE.md` (tabla por etapa).

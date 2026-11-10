# ArCH: resumen antes de usarlo

**Pregunta.** Qué hay en ArCH_labelled: cuántos puntos por escena, tamaño físico y qué clases aparecen en cuántas escenas. Con eso se decide qué clases usar y con qué entrenar (todo o parte).

- `resumir_arch.py`: lee los `.txt` por bloques y guarda un json por escena (reanudable), `resumen_arch.json` y `resumen_arch.md`.
- Formato: `x y z r g b label Nx Ny Nz`. Clases: 0 arch, 1 column, 2 moldings, 3 floor, 4 door_window, 5 wall, 6 stairs, 7 vault, 8 roof, 9 other.
- Colab: salida en `PFC1_HyperNCD_Sonata/arch/resumen/` (Drive). Sin GPU.
- Licencia CC BY-NC-SA 4.0 (Politecnico di Torino); citar el dataset en la tesis.

Resultados: pendiente de correr.

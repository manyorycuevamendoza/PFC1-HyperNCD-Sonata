# Etiquetar el recorte de PL13 en CloudCompare

Archivo: `PL13_recorte_para_etiquetar.ply` (214 645 puntos, centro de la nave, ±7 m).
Cada punto trae un campo escalar `idx` (índice del punto en el `.npy`). **No lo borres.**
Con ese campo yo reconstruyo las etiquetas, así que no hay que tocar valores de clase.

## Método: un archivo por clase

Guardas cada clase en un archivo aparte con el nombre indicado. Los puntos que no
guardes quedan sin etiqueta y se ignoran.

| archivo a guardar | qué es |
|---|---|
| `clase_1_suelo.ply` | suelo |
| `clase_2_muro.ply` | muros |
| `clase_3_columna.ply` | columnas / pilares |
| `clase_4_boveda.ply` | bóveda / cúpula / techo |
| `clase_5_otro.ply` | altar, bancas, puertas, objetos (opcional) |

## Pasos

1. Abre CloudCompare y arrastra `PL13_recorte_para_etiquetar.ply`. Selecciona la nube en
   el panel izquierdo (*DB Tree*). En *Properties > Colors* elige RGB.
2. Herramienta **Segment** (tijeras, tecla `T`). Con clic izquierdo dibujas un polígono
   alrededor de lo que quieres; clic derecho cierra el polígono.
3. Pulsa **Segment In** y luego **Confirm** (la marca verde). La nube se parte en dos:
   lo que encerraste y el resto.
4. Selecciona la parte que encerraste (por ejemplo, el suelo) → **File > Save** →
   formato PLY → nombre `clase_1_suelo.ply` en esta misma carpeta.
5. Vuelve a seleccionar la parte restante y repite desde el paso 2 con la siguiente clase.
   Empieza por lo grande y claro (suelo, bóveda, muros); lo difícil queda al final.
6. Para cada polígono de la misma clase puedes guardar varios archivos
   (`clase_2_muro_a.ply`, `clase_2_muro_b.ply`): el nombre solo necesita empezar por
   `clase_N_`.

Consejo: gira la vista de lado (tecla `7` o los botones de vistas) para recortar
el suelo con un polígono casi horizontal, y la vista de planta para los muros.

Cuando termines, avísame. Con suelo, muros y bóveda ya hay 3 clases y un mIoU útil.
No hace falta etiquetar el 100 % de los puntos.

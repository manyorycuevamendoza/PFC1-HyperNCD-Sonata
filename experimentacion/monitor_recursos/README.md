# Monitor de recursos (tiempo, RAM, GPU)

Corre cualquier comando y mide, mientras dura: tiempo total, pico de RAM del proceso y del sistema, memoria y uso de GPU, uso de CPU.

`python monitor_recursos.py --salida DIR [--cada 5] [--nombre etiqueta] -- <comando>`

Guarda `recursos.csv`, `resumen.json` y `recursos.png` en DIR. Mide el proceso lanzado y sus hijos (sirve con el Python de `hyperncd-env`).
Sin probar en Colab todavía. Prueba local: `python monitor_recursos.py --salida /tmp/x -- python -c "import time; time.sleep(3)"`.

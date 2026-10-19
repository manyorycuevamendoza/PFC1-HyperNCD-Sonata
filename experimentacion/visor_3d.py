"""Visor 3D del proyecto: un escaneo en todas sus etapas, en una sola escena.

Genera un .html (Plotly, se abre con doble clic) con una escena 3D y un boton
por etapa. Todas las etapas comparten el mismo sistema de coordenadas, la misma
escala y la MISMA CAMARA: al cambiar de boton el angulo no se mueve. Asi se
gira una vez, se elige el angulo y se baja un PNG de cada etapa con el boton de
la camara de fotos (arriba a la derecha), con encuadre identico.

Botones (aparecen segun los archivos que se pasen):
    Crudo                    --las + --json   entrada del Modulo 1
    M1: color real           --npy            salida del Modulo 1 (matriz S)
    M1: normales             --npy            normal de cada punto como color
    M3: PCA de f_ssl         --feats          mismo color = descriptores parecidos
    M3: similitud con ◆      --feats          rojo = parecido al punto ◆
    M3: k-means              --feats          grupos sin etiquetas

Uso (desde la raiz del repo):
    python experimentacion/visor_3d.py \
        --las   experimentacion/datos/modulo3_PL13/PL13_SCN0001.las \
        --npy   experimentacion/datos/modulo3_PL13/PL13_SCN0001_preprocesado.npy \
        --json  experimentacion/datos/modulo3_PL13/PL13_SCN0001_preprocesado.json \
        --feats experimentacion/datos/modulo3_PL13/PL13_feats.npz \
        --output experimentacion/resultados/visor_PL13.html

Angulo inicial: --camara X,Y,Z es la posicion del ojo (default 1.25,-1.25,0.95,
vista oblicua desde arriba). Ejemplos: 0,0,2.2 = planta desde arriba;
2.2,0,0.3 = de costado. Requiere: numpy, plotly; laspy si se usa --las.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npy", required=True, help="salida de preprocesar.py (matriz S)")
    ap.add_argument("--json", help="metadatos de preprocesar.py (offset y conteos)")
    ap.add_argument("--las", help=".las original (requiere --json)")
    ap.add_argument("--feats", help=".npz de extraer_features.py")
    ap.add_argument("--output", required=True, help="ruta del .html")
    ap.add_argument("--camara", default="1.25,-1.25,0.95",
                    help="posicion del ojo X,Y,Z (default 1.25,-1.25,0.95)")
    ap.add_argument("--puntos", type=int, default=100_000,
                    help="puntos por etapa (submuestreo, default 100000)")
    ap.add_argument("--tam", type=float, default=1.5, help="tamano del punto")
    ap.add_argument("--k", type=int, default=8, help="grupos de k-means")
    ap.add_argument("--consulta", type=int, default=-1,
                    help="punto de consulta de la similitud (-1 = al azar)")
    ap.add_argument("--recorte", help="X0,Y0,Z0,LADO: dibuja solo ese cubo, con todos "
                    "sus puntos (si no superan --puntos)")
    ap.add_argument("--brillo", type=float, default=1.0,
                    help="multiplica el color real (ej. 2.2 para escaneos oscuros)")
    ap.add_argument("--semilla", type=int, default=0)
    return ap.parse_args()


def normalizar(x):
    lo, hi = np.percentile(x, [2, 98], axis=0)
    return np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)


def a_texto_rgb(c):
    c = (np.clip(c, 0, 1) * 255).astype(int)
    return [f"rgb({r},{g},{b})" for r, g, b in c]


def coolwarm(v):
    """v en [-1, 1] -> azul (distinto) / blanco / rojo (parecido)."""
    t = (np.clip(v, -1, 1) + 1) / 2
    azul, blanco, rojo = np.array([.23, .30, .75]), np.ones(3), np.array([.71, .02, .15])
    return np.where(t[:, None] < .5,
                    azul + (blanco - azul) * (t[:, None] * 2),
                    blanco + (rojo - blanco) * ((t[:, None] - .5) * 2))


TAB10 = np.array([[31, 119, 180], [255, 127, 14], [44, 160, 44], [214, 39, 40],
                  [148, 103, 189], [140, 86, 75], [227, 119, 194], [127, 127, 127],
                  [188, 189, 34], [23, 190, 207]]) / 255


def descriptores(feats_path, S, rng, k, consulta):
    """PCA, similitud y k-means de f_ssl, llevados a cada punto de S."""
    z = np.load(feats_path)
    feat, inverse = z["feat"].astype(np.float32), z["inverse"]
    # El .npz solo sirve con el MISMO .npy que entro a Sonata, en el mismo orden
    # de filas: dos corridas del Modulo 1 con distinta version de Open3D dan los
    # mismos puntos en otro orden. Se comprueba que cada punto de S caiga en su
    # celda (Sonata centra x,y en el medio de la caja y z en el minimo).
    if len(inverse) != len(S):
        raise ValueError("el .npz no corresponde a este .npy (distinto numero de puntos)")
    mn, mx = S[:, :3].min(0), S[:, :3].max(0)
    centro = np.array([(mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, mn[2]])
    lado = float(z["grid_size"])
    if np.median(np.abs(S[:, :3] - centro - z["coord"][inverse]).max(1)) > lado:
        raise ValueError("el .npz no corresponde a este .npy: mismos puntos en otro "
                         "orden. Usa el .npy exacto que entro a extraer_features.py")
    m = rng.choice(len(feat), min(len(feat), 100_000), replace=False)
    media = feat[m].mean(0)
    _, _, vt = np.linalg.svd(feat[m] - media, full_matrices=False)
    pca = normalizar((feat - media) @ vt[:3].T)
    fn = feat - media
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-8
    q = consulta if consulta >= 0 else int(rng.integers(len(feat)))
    sim = fn @ fn[q]
    centros = fn[m][rng.choice(len(m), k, replace=False)]
    for _ in range(30):
        g = (fn[m] @ centros.T - .5 * (centros ** 2).sum(1)).argmax(1)
        for c in range(k):
            if (g == c).any():
                centros[c] = fn[m][g == c].mean(0)
    grupo = np.concatenate([(fn[i:i + 200_000] @ centros.T
                             - .5 * (centros ** 2).sum(1)).argmax(1)
                            for i in range(0, len(fn), 200_000)])
    # punto de S mas cercano a la celda de consulta, para dibujar el rombo
    q_xyz = S[np.flatnonzero(inverse == q)[0], :3]
    return pca[inverse], sim[inverse], grupo[inverse], q_xyz


def main():
    args = parse_args()
    rng = np.random.default_rng(args.semilla)
    meta = json.load(open(args.json)) if args.json else {}
    S = np.load(args.npy)
    caja = None
    if args.recorte:
        x0, y0, z0, lado_c = map(float, args.recorte.split(","))
        caja = np.array([[x0, x0 + lado_c], [y0, y0 + lado_c], [z0, z0 + lado_c]])

    def elegir(xyz):
        """Indices a dibujar: los del recorte (si hay) y como mucho --puntos."""
        idx = np.arange(len(xyz))
        if caja is not None:
            dentro = np.all((xyz >= caja[:, 0]) & (xyz < caja[:, 1]), axis=1)
            idx = np.flatnonzero(dentro)
        if len(idx) > args.puntos:
            idx = rng.choice(idx, args.puntos, replace=False)
        return idx

    j = elegir(S[:, :3])

    etapas = []          # (nombre, xyz, colores)
    if args.las:
        import laspy
        las = laspy.read(args.las)
        # float32, como la matriz S: asi el recorte decide los bordes igual
        xyz = (np.stack([las.x, las.y, las.z], 1)
               - np.array(meta["offset_utm"])).astype(np.float32)
        i = elegir(xyz)
        xyz = xyz[i]
        rgb = np.stack([las.red[i], las.green[i], las.blue[i]], 1).astype(np.float32)
        rgb /= 65535.0 if rgb.max() > 255 else 255.0
        etapas.append((f"Crudo ({len(i):,})", xyz, rgb * args.brillo))
    if caja is not None:
        print(f"recorte: crudo {len(etapas[0][1]) if args.las else '-'}  "
              f"preprocesado {len(j)}")
    etapas.append((f"M1: color real ({len(j):,})", S[j, :3], S[j, 3:6] / 255 * args.brillo))
    etapas.append(("M1: normales", S[j, :3], (S[j, 6:9] + 1) / 2))
    q_xyz = None
    if args.feats:
        pca, sim, grupo, q_xyz = descriptores(args.feats, S, rng, args.k, args.consulta)
        etapas.append(("M3: PCA de f_ssl", S[j, :3], pca[j]))
        etapas.append(("M3: similitud con ◆", S[j, :3], coolwarm(sim[j])))
        etapas.append((f"M3: k-means (k={args.k})", S[j, :3], TAB10[grupo[j] % 10]))

    trazas = [go.Scatter3d(x=x[:, 0], y=x[:, 1], z=x[:, 2], mode="markers",
                           marker=dict(size=args.tam, color=a_texto_rgb(c)),
                           hoverinfo="skip", name=n, visible=(k == 0))
              for k, (n, x, c) in enumerate(etapas)]
    if q_xyz is not None:
        trazas.append(go.Scatter3d(x=[q_xyz[0]], y=[q_xyz[1]], z=[q_xyz[2]],
                                   mode="markers", name="◆", visible=False,
                                   marker=dict(size=9, color="black", symbol="diamond")))
    if caja is not None:   # aristas del cubo, siempre visibles
        v = np.array([[caja[0, x], caja[1, y], caja[2, z]]
                      for x in (0, 1) for y in (0, 1) for z in (0, 1)])
        aristas = [(p, q) for p in range(8) for q in range(p + 1, 8)
                   if (np.abs(v[p] - v[q]) > 0).sum() == 1]
        xs, ys, zs = [], [], []
        for p, q in aristas:
            for w in (v[p], v[q], [None] * 3):
                xs.append(w[0]); ys.append(w[1]); zs.append(w[2])
        trazas.append(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", name="cubo",
                                   line=dict(color="gray", width=2), hoverinfo="skip"))
    n_et = len(etapas)

    def visibles(k):
        v = [e == k for e in range(n_et)]
        if q_xyz is not None:
            v.append(etapas[k][0].startswith("M3: similitud"))
        if caja is not None:
            v.append(True)
        return v

    botones = [dict(label=n, method="update", args=[{"visible": visibles(k)}])
               for k, (n, _, _) in enumerate(etapas)]

    # escala fija (la de S) para que cambiar de etapa no reencuadre
    lim = caja if caja is not None else \
        np.array([np.percentile(S[:, d], [0.5, 99.5]) for d in range(3)])
    lado = lim[:, 1] - lim[:, 0]
    ojo = dict(zip("xyz", map(float, args.camara.split(","))))
    nombre = Path(args.npy).stem.replace("_preprocesado", "")
    pie = ""
    if meta:
        pie = (f"Crudo {meta['n_crudo']:,} → SOR {meta['n_sor']:,} → vóxel "
               f"{meta['voxel'] * 100:g} cm: {meta['n_voxel']:,} puntos.  "
               f""
               + ("Recorte: se dibujan todos los puntos del cubo." if caja is not None
                  else f"Cada etapa dibuja {args.puntos:,} puntos al azar."))

    fig = go.Figure(trazas)
    fig.update_layout(
        title=dict(text=f"{nombre}: gira con el mouse · el ángulo se mantiene al "
                        f"cambiar de etapa · 📷 arriba a la derecha baja el PNG", y=0.98),
        showlegend=False, height=850, uirevision="fija",
        scene=dict(camera=dict(eye=ojo, up=dict(x=0, y=0, z=1)),
                   aspectmode="manual",
                   aspectratio=dict(zip("xyz", (lado / lado.max()).tolist())),
                   xaxis=dict(visible=False, range=lim[0].tolist()),
                   yaxis=dict(visible=False, range=lim[1].tolist()),
                   zaxis=dict(visible=False, range=lim[2].tolist())),
        updatemenus=[dict(type="buttons", direction="right", x=0.5, y=1.0,
                          xanchor="center", yanchor="bottom", buttons=botones)],
        annotations=[dict(text=pie, showarrow=False, x=0.5, y=0, xref="paper",
                          yref="paper", font=dict(size=13))],
        margin=dict(l=0, r=0, t=110, b=30))

    salida = Path(args.output)
    salida.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(salida, include_plotlyjs="cdn", config=dict(
        displaylogo=False,
        toImageButtonOptions=dict(format="png", filename=f"{nombre}_vista",
                                  scale=2)))
    print(f"[ok] {salida}")


if __name__ == "__main__":
    main()

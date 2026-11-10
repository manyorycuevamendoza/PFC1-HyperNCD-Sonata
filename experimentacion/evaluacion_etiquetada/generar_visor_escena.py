"""Genera UN visor HTML de la escena etiquetada, de M1 a M4, con aciertos y errores (TP / FP / FN) y metricas.

Vistas (cada pestaña):
  M1        color real, normal vertical, clase real
  M2        cabeza de HyperNCD, novedoso/conocido, clase predicha, aciertos de la clase elegida (punto a punto)
  M3 y M4   grupos de cada modulo, variante de M3 (original/adaptado) y K   -- necesitan los grupos por punto (ver abajo)
  Errores   verde = acierto (TP), naranja = falso negativo (FN), rojo = falso positivo (FP) para la clase elegida
  Comparar  tabla original vs adaptado por modulo y K (todas las clases y sin `otro`)
Cada pestaña muestra al lado las metricas de lo que se ve (mIoU, mAcc, allAcc, ARI, NMI, TP/FP/FN por clase, confusion).

Datos pesados: se muestrean --puntos puntos (120 000 por defecto) y todo va en binario base64 dentro del HTML (unos pocos MB),
asi abre en equipos con poca RAM. Plotly se carga de internet (CDN), como los demas visores.

Los grupos por punto de M4 salen de `m4_<variante>/grupos_PL13.npz` (los escribe correr_modulo4_real.py). Si no estan, las
pestañas M3, M4 y Errores de M3/M4 avisan y el resto funciona igual.

Uso:
    python experimentacion/evaluacion_etiquetada/generar_visor_escena.py \
        --datos experimentacion/escena_capilla_etiquetada/datos \
        --resultados experimentacion/escena_capilla_etiquetada/resultados \
        --salida experimentacion/escena_capilla_etiquetada/visor/visor_escena.html
"""
import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

MODULOS = ["M2", "M3", "M4_base", "M4"]
COLORES = {"arco": "#e6194b", "columna": "#3cb44b", "cupula": "#0082c8", "pared": "#f58231", "techo": "#911eb4", "otro": "#aaaaaa"}


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def asignaciones(T, clases, ignorar=None):
    """grupo -> clase (indices de `clases`, -1 = sin clase) por mayoria y por emparejamiento hungaro."""
    T = np.asarray(T, np.int64).copy()
    if ignorar:
        T[:, clases.index(ignorar)] = 0
    may = np.where(T.sum(1) > 0, T.argmax(1), -1)
    fila, col = linear_sum_assignment(-T)
    uno = np.full(T.shape[0], -1)
    uno[fila] = col
    uno[T.sum(1) == 0] = -1
    return {"mayoria": may.tolist(), "uno_a_uno": uno.tolist()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datos", required=True)
    ap.add_argument("--resultados", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--escena", default="capilla_etiquetada")
    ap.add_argument("--puntos", type=int, default=120_000)
    ap.add_argument("--grupos", nargs="*", default=[], help="nombre=ruta/grupos_PL13.npz (si no se dan, se buscan en datos/)")
    a = ap.parse_args()
    D, R, E = Path(a.datos), Path(a.resultados), a.escena
    rng = np.random.default_rng(0)

    S = np.load(D / f"m1/salida/{E}/{E}.npy", mmap_mode="r")
    lab = np.load(D / f"m1/salida/{E}/{E}_etiquetas.npy")
    info = json.loads((D / f"m1/salida/{E}/{E}_etiquetas.json").read_text(encoding="utf-8"))
    clases = info["clases"]
    N = len(lab)
    sel = np.sort(rng.choice(N, min(N, a.puntos), replace=False))
    xyz = np.asarray(S[sel, :3], np.float32)
    xyz -= xyz.min(0)
    rgb = np.clip(np.asarray(S[sel, 3:6], np.float32), 0, 255).astype(np.uint8)
    nz = (np.clip(np.abs(np.asarray(S[sel, 8], np.float32)), 0, 1) * 255).astype(np.uint8)
    gt = np.where(lab[sel] >= 0, lab[sel], 255).astype(np.uint8)

    z = np.load(D / f"m2/salida/{E}/{E}_fsem.npz")
    inv = z["inverse"][sel]
    cabeza = z["pred_label"][inv].astype(np.uint8)
    novel = (z["es_novedosa"][inv] == 1).astype(np.uint8)
    et = json.loads((R / "etapas/etapas.json").read_text(encoding="utf-8"))
    nl = et["M2"]["cabezas"]["conocidas"]
    T2 = np.array(et["M2"]["tabla_cabeza_x_clase"], np.int64)
    asig_m2 = {"todas": asignaciones(T2, clases), "sin_otro": asignaciones(T2, clases, "otro")}

    ev = {"todas": json.loads((R / "evaluacion/todas/evaluacion.json").read_text(encoding="utf-8")),
          "sin_otro": json.loads((R / "evaluacion/sin_otro/evaluacion.json").read_text(encoding="utf-8"))}
    variantes = list(ev["todas"]["variantes"])

    # ---- grupos por punto de M4 (opcional) ----
    rutas = dict(x.split("=", 1) for x in a.grupos)
    if not rutas:
        for v in variantes:
            p = D / f"m4/salida/{E}/m4_{v}/grupos_PL13.npz"
            if p.exists():
                rutas[v] = str(p)
    grupos, asig_g = {}, {}
    for v, ruta in rutas.items():
        g = np.load(ruta)
        for clave in g.files:
            if not clave.startswith("K"):
                continue
            K, mod = clave[1:].split("_", 1)
            full = np.full(N, 255, np.uint8)
            full[g["idx_puntos"]] = g[clave][g["region"]]
            ident = f"{v}|{K}|{mod}"
            grupos[ident] = b64(full[sel])
            Tg = np.array(ev["todas"]["variantes"][v]["K"][K][mod]["cruce_grupos_x_clases"], np.int64)
            asig_g[ident] = {"todas": asignaciones(Tg, clases), "sin_otro": asignaciones(Tg, clases, "otro")}

    datos = {"escena": E, "clases": clases, "colores": [COLORES.get(c, "#888888") for c in clases], "nl": nl,
             "n": int(len(sel)), "N": int(N), "variantes": variantes,
             "xyz": b64(xyz), "rgb": b64(rgb), "nz": b64(nz), "gt": b64(gt), "cabeza": b64(cabeza), "novel": b64(novel),
             "grupos": grupos, "asig_g": asig_g, "asig_m2": asig_m2, "eval": ev, "etapas": et}
    html = PLANTILLA.replace("__TITULO__", f"Escena {E}").replace("__DATOS__", json.dumps(datos, ensure_ascii=False, separators=(",", ":")))
    sal = Path(a.salida)
    sal.parent.mkdir(parents=True, exist_ok=True)
    sal.write_text(html, encoding="utf-8")
    print(f"{sal}  ({sal.stat().st_size / 2**20:.1f} MB; {len(sel):,} de {N:,} puntos; vistas de grupos: {len(grupos)})")
    if not grupos:
        print("AVISO: no hay grupos por punto de M4; las pestañas M3/M4 solo mostraran metricas.")


PLANTILLA = r"""<!DOCTYPE html><html lang="es"><head><meta charset="utf-8"><title>__TITULO__ · M1 a M4</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
 body{margin:0;font:13px/1.4 system-ui,Segoe UI,sans-serif;color:#1c2430;background:#f4f6f9;display:flex;height:100vh}
 aside{width:430px;overflow:auto;padding:12px 14px;background:#fff;border-right:1px solid #dde2e8}
 main{flex:1;min-width:0}#g{width:100%;height:100vh}
 h1{font-size:15px;margin:0 0 8px} h3{font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:#667085;margin:14px 0 5px}
 .tabs{display:flex;flex-wrap:wrap;gap:4px}.tabs button{border:0;border-radius:6px;padding:6px 10px;background:#e8ecf1;cursor:pointer;font-weight:600}
 .tabs button.on{background:#2563eb;color:#fff}
 label{display:block;margin:5px 0 2px;color:#475467} select,input[type=range]{width:100%}
 table{border-collapse:collapse;width:100%;font-size:12px}td,th{padding:3px 5px;border-bottom:1px solid #eef0f3;text-align:right}th:first-child,td:first-child{text-align:left}
 .leyenda span{display:inline-block;margin:2px 8px 2px 0}.leyenda i{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:4px;vertical-align:-1px}
 .aviso{background:#fff8e1;border:1px solid #f0d78a;border-radius:6px;padding:7px 9px;margin:6px 0}
 .malo{color:#b42318;font-weight:600}.ok{color:#067647;font-weight:600}.sup{color:#667085;font-size:11px}
</style></head><body>
<aside id="panel">
 <h1>__TITULO__ · de M1 a M4</h1>
 <div class="tabs" id="tabs"></div>
 <div id="ctl"></div>
 <h3>Leyenda</h3><div class="leyenda" id="ley"></div>
 <h3>Métricas de lo que ves</h3><div id="met"></div>
</aside><main><div id="g"></div></main>
<script>
const D = __DATOS__;
const dec = (s, T) => { const b = atob(s), u = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return T ? new T(u.buffer) : u; };
const n = D.n, XYZ = dec(D.xyz, Float32Array), RGB = dec(D.rgb), NZ = dec(D.nz), GT = dec(D.gt), CAB = dec(D.cabeza), NOV = dec(D.novel);
const GR = {}; for (const k in D.grupos) GR[k] = dec(D.grupos[k]);
const CL = D.clases, C = CL.length, COL = D.colores, OTRO = CL.indexOf("otro");
const PAL = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf","#aec7e8","#ffbb78","#98df8a","#c5b0d5"];
const GRIS = "#d9d9d9", VERDE = "#2ca02c", NARANJA = "#ff7f0e", ROJO = "#d62728", FONDO = "#e6e6e6";
const S = {tab: "m1", vista: "rgb", modo: "todas", asig: "uno_a_uno", variante: D.variantes[0], K: null, mod: "M4", clase: CL[0] === "otro" ? CL[1] : CL[0], fuente: "m2", sinotro: false, tam: 1.6};
const hex = (r, g, b) => "#" + [r, g, b].map(v => v.toString(16).padStart(2, "0")).join("");
const Ks = [...new Set(Object.keys(D.grupos).map(k => k.split("|")[1]))].sort((a, b) => a - b); S.K = Ks.length ? Ks[Math.min(1, Ks.length - 1)] : "8";
const hayGrupos = Object.keys(D.grupos).length > 0;
const f3 = x => (x === null || x === undefined) ? "-" : (100 * x).toFixed(1);
const fmt = x => (x === null || x === undefined) ? "-" : x.toFixed(3);

function clasePred() {   // clase predicha por punto (255 = sin clase) segun la fuente elegida
  const p = new Uint8Array(n).fill(255);
  if (S.fuente === "m2") { const as = D.asig_m2[S.modo][S.asig]; for (let i = 0; i < n; i++) { const c = as[CAB[i]]; p[i] = c >= 0 ? c : 255; } return p; }
  const id = S.variante + "|" + S.K + "|" + S.mod; const g = GR[id]; if (!g) return null; const as = D.asig_g[id][S.modo][S.asig];
  for (let i = 0; i < n; i++) { if (g[i] === 255) continue; const c = as[g[i]]; p[i] = c >= 0 ? c : 255; } return p;
}
function colores() {
  const col = new Array(n), leyenda = [];
  const v = S.vista;
  if (v === "rgb") { for (let i = 0; i < n; i++) col[i] = hex(RGB[3*i], RGB[3*i+1], RGB[3*i+2]); }
  else if (v === "nz") { for (let i = 0; i < n; i++) { const t = NZ[i]; col[i] = hex(t, 80 + (t >> 1), 255 - t); } leyenda.push(["#0050ff", "normal horizontal"], ["#ff80c0", "normal vertical"]); }
  else if (v === "gt") { for (let i = 0; i < n; i++) col[i] = GT[i] === 255 ? FONDO : COL[GT[i]]; CL.forEach((c, i) => leyenda.push([COL[i], c])); }
  else if (v === "cabeza") { for (let i = 0; i < n; i++) col[i] = PAL[CAB[i]]; for (let h = 0; h < 13; h++) leyenda.push([PAL[h], "cabeza " + h + (h >= D.nl ? " (novedosa)" : " (conocida)")]); }
  else if (v === "novel") { for (let i = 0; i < n; i++) col[i] = NOV[i] ? "#e07b39" : "#2b7bba"; leyenda.push(["#e07b39", "novedoso"], ["#2b7bba", "conocido"]); }
  else if (v === "pred") { const p = clasePred(); if (!p) return null; for (let i = 0; i < n; i++) col[i] = p[i] === 255 ? FONDO : COL[p[i]]; CL.forEach((c, i) => leyenda.push([COL[i], "predicha: " + c])); }
  else if (v === "grupos") { const g = GR[S.variante + "|" + S.K + "|" + S.mod]; if (!g) return null; for (let i = 0; i < n; i++) col[i] = g[i] === 255 ? FONDO : PAL[g[i] % PAL.length]; for (let k = 0; k < +S.K; k++) leyenda.push([PAL[k % PAL.length], "grupo " + k]); }
  else if (v === "acierto" || v === "clase") { const p = clasePred(); if (!p) return null;
    if (v === "acierto") { for (let i = 0; i < n; i++) col[i] = GT[i] === 255 ? FONDO : (GT[i] === p[i] ? VERDE : ROJO); leyenda.push([VERDE, "bien clasificado"], [ROJO, "mal clasificado"], [FONDO, "sin etiqueta / ignorado"]); }
    else { const c = CL.indexOf(S.clase); for (let i = 0; i < n; i++) { const g = GT[i] === c, q = p[i] === c; col[i] = g && q ? VERDE : g ? NARANJA : q ? ROJO : GRIS; }
      leyenda.push([VERDE, "TP: de la clase y predicho bien"], [NARANJA, "FN: es de la clase y se predijo otra cosa"], [ROJO, "FP: se predijo la clase y no lo es"], [GRIS, "el resto"]); } }
  return {col, leyenda};
}
let ocultar = false;
function dibujar() {
  const r = colores(); const el = document.getElementById("g");
  if (!r) { document.getElementById("ley").innerHTML = "<div class='aviso'>No hay grupos por punto de M3/M4 en este visor (faltan <code>m4_&lt;variante&gt;/grupos_PL13.npz</code>). Las métricas de la derecha sí están.</div>"; return; }
  document.getElementById("ley").innerHTML = r.leyenda.map(([c, t]) => `<span><i style="background:${c}"></i>${t}</span>`).join("");
  const idx = []; for (let i = 0; i < n; i++) if (!(ocultar && GT[i] === OTRO)) idx.push(i);
  const x = idx.map(i => XYZ[3*i]), y = idx.map(i => XYZ[3*i+1]), z = idx.map(i => XYZ[3*i+2]), c = idx.map(i => r.col[i]);
  Plotly.react(el, [{type: "scatter3d", mode: "markers", x, y, z, marker: {size: S.tam, color: c}, hoverinfo: "skip"}],
    {margin: {l: 0, r: 0, t: 0, b: 0}, uirevision: "fija", scene: {aspectmode: "data", xaxis: {visible: false}, yaxis: {visible: false}, zaxis: {visible: false},
      camera: {eye: {x: 1.25, y: -1.25, z: 0.95}, up: {x: 0, y: 0, z: 1}}}}, {displaylogo: false, responsive: true});
}
const sel = (id, opts, val, fn) => `<label>${id}</label><select onchange="${fn}(this.value)">` + opts.map(([v, t]) => `<option value="${v}" ${v == val ? "selected" : ""}>${t}</option>`).join("") + "</select>";
const VISTAS = {m1: [["rgb", "Color real"], ["nz", "Normal vertical"], ["gt", "Clase real (etiqueta)"]],
  m2: [["cabeza", "Cabeza predicha por HyperNCD"], ["novel", "Novedoso / conocido"], ["pred", "Clase predicha (cabeza → clase)"], ["clase", "TP / FP / FN de una clase"]],
  m3: [["grupos", "Grupos de M3 (solo f_ssl)"], ["pred", "Clase predicha"], ["acierto", "Aciertos"], ["clase", "TP / FP / FN de una clase"]],
  m4: [["grupos", "Grupos del módulo"], ["pred", "Clase predicha"], ["acierto", "Aciertos"], ["clase", "TP / FP / FN de una clase"]]};
function controles() {
  let h = "";
  if (S.tab === "comparar") { document.getElementById("ctl").innerHTML = sel("Evaluación", [["todas", "Todas las clases"], ["sin_otro", "Sin la clase «otro»"]], S.modo, "cambiar('modo')"); return; }
  h += sel("Vista", VISTAS[S.tab], S.vista, "cambiar('vista')");
  if (S.tab === "m2") S.fuente = "m2"; else { S.fuente = "g"; if (S.tab === "m3") S.mod = "M3";
    h += sel("M3 usado", D.variantes.map(v => [v, v]), S.variante, "cambiar('variante')") + sel("K (grupos)", (Ks.length ? Ks : ["8"]).map(k => [k, k]), S.K, "cambiar('K')");
    if (S.tab === "m4") h += sel("Módulo", [["M4", "M4 (propuesta)"], ["M4_base", "M4 base"], ["M2", "M2 por regiones"]], S.mod, "cambiar('mod')"); }
  if (["pred", "acierto", "clase"].includes(S.vista)) h += sel("Asignar grupos a clases", [["uno_a_uno", "1 a 1 (húngaro)"], ["mayoria", "Por mayoría"]], S.asig, "cambiar('asig')") + sel("Evaluación", [["todas", "Todas las clases"], ["sin_otro", "Sin «otro»"]], S.modo, "cambiar('modo')");
  if (S.vista === "clase") h += sel("Clase", CL.map(c => [c, c]), S.clase, "cambiar('clase')");
  h += `<label><input type="checkbox" ${ocultar ? "checked" : ""} onchange="ocultar=this.checked;dibujar()"> Ocultar los puntos de «otro»</label>`;
  h += `<label>Tamaño del punto</label><input type="range" min="0.5" max="5" step="0.1" value="${S.tam}" oninput="S.tam=+this.value;dibujar()">`;
  document.getElementById("ctl").innerHTML = h;
}
function setS(k, v) { S[k] = v; refrescar(); }
window.cambiar = k => v => setS(k, v);     // los selects llaman a cambiar('clave')(valor)

function tablaClases(pc) {
  let h = "<table><tr><th>Clase</th><th>Reales</th><th>TP</th><th>FP</th><th>FN</th><th>IoU</th><th>Prec.</th><th>Recall</th><th>F1</th><th>FPR</th><th>FNR</th></tr>";
  for (const c in pc) { const d = pc[c]; if (!d.puntos_reales) continue;
    h += `<tr><td>${c}</td><td>${d.puntos_reales.toLocaleString()}</td><td>${d.TP.toLocaleString()}</td><td>${d.FP.toLocaleString()}</td><td>${d.FN.toLocaleString()}</td><td>${f3(d.IoU)}</td><td>${f3(d.precision)}</td><td>${f3(d.recall)}</td><td>${f3(d.F1)}</td><td>${f3(d.FPR)}</td><td>${f3(d.FNR)}</td></tr>`; }
  return h + "</table><div class='sup'>Todo en %, salvo los conteos. FP = se predijo la clase y no lo es; FN = es de la clase y se predijo otra.</div>";
}
function tablaConfusion(r) {
  const cols = r.columnas_confusion, M = r.confusion_clases; let h = "<table><tr><th>real ↓ / pred →</th>" + cols.map(c => `<th>${c}</th>`).join("") + "</tr>";
  M.forEach((fila, i) => { const t = fila.reduce((a, b) => a + b, 0) || 1; h += `<tr><td>${CLd(i)}</td>` + fila.map((v, j) => { const p = v / t; return `<td style="background:rgba(37,99,235,${Math.min(p, 1) * 0.85})">${p >= 0.005 ? (100 * p).toFixed(0) : ""}</td>`; }).join("") + "</tr>"; });
  return h + "</table><div class='sup'>% de cada clase real (fila).</div>";
}
const CLd = i => (D.eval[S.modo].clases || CL)[i];
function resumen(r, nombre) {
  const m = r[S.asig] ? r : null; const x = r[S.asig === "mayoria" ? "mayoria" : "uno_a_uno"];
  return `<b>${nombre}</b><table><tr><th>mIoU</th><th>mAcc</th><th>allAcc</th><th>ARI</th><th>NMI</th></tr><tr><td>${f3(x.mIoU)}</td><td>${f3(x.mAcc)}</td><td>${f3(x.allAcc)}</td><td>${fmt(r.ARI)}</td><td>${fmt(r.NMI)}</td></tr></table>`;
}
function metricas() {
  const el = document.getElementById("met"), E = D.etapas; let h = "";
  if (S.tab === "m1") { const m = E.M1; h += `<table><tr><th>Puntos</th><th>Cantidad</th></tr><tr><td>crudo</td><td>${m.puntos.crudo.toLocaleString()}</td></tr><tr><td>tras SOR (−${m.puntos.SOR_descarto_pct.toFixed(1)} %)</td><td>${m.puntos.tras_SOR.toLocaleString()}</td></tr><tr><td>tras voxel 5 cm</td><td>${m.puntos.tras_voxel.toLocaleString()}</td></tr></table>`;
    h += "<h3>Clases antes y después</h3><table><tr><th>Clase</th><th>Crudo</th><th>Tras M1</th><th>Parte antes</th><th>Parte después</th></tr>" + Object.entries(m.por_clase).map(([c, d]) => `<tr><td>${c}</td><td>${d.crudo.toLocaleString()}</td><td>${d.tras_M1.toLocaleString()}</td><td>${d.parte_antes_pct.toFixed(1)}</td><td>${d.parte_despues_pct.toFixed(1)}</td></tr>`).join("") + "</table>";
    h += `<div class="sup">Etiquetas: mediana ${m.etiquetas.dist_mediana_cm.toFixed(1)} cm al punto crudo; sin etiqueta: ${m.etiquetas.sin_etiqueta}. Normales: ${m.normales.norma_unitaria_pct.toFixed(0)} % unitarias. Si la «parte» de una clase cambia mucho, M1 la sesga.</div>`; }
  else if (S.tab === "m2") { const m = E.M2, c = m.como_clasificador_punto_a_punto[S.modo];
    h += `<div class="aviso"><b>${m.cabeza_dominante.pct.toFixed(1)} %</b> de los puntos caen en la <b>cabeza ${m.cabeza_dominante.cabeza}</b>; ${m.puntos_novedosos_pct.toFixed(1)} % se marcan como novedosos. Rango efectivo de f_sem: <b>${m.f_sem.rango_efectivo.toFixed(1)}</b> (el de Sonata ronda 100): el descriptor está casi colapsado.</div>`;
    h += resumen({mayoria: c.mayoria, uno_a_uno: c.uno_a_uno, ARI: c.ARI, NMI: c.NMI}, "M2 como clasificador punto a punto") + tablaClases(c[S.asig].por_clase) + "<h3>Confusión</h3>" + tablaConfusion(c[S.asig]); }
  else if (S.tab === "comparar") { const d = D.eval[S.modo]; const Kl = Object.keys(d.variantes[D.variantes[0]].K);
    h += `<div class="sup">mIoU % con asignación por mayoría / 1 a 1. Referencia (todo a «${d.referencia_todo_a_la_mayoritaria.clase}»): ${f3(d.referencia_todo_a_la_mayoritaria.mIoU)}.</div><table><tr><th>K · Módulo</th>` + D.variantes.map(v => `<th>${v}</th>`).join("") + "</tr>";
    for (const K of Kl) for (const mod of ["M2", "M3", "M4_base", "M4"]) h += `<tr><td>${K} · ${mod}</td>` + D.variantes.map(v => { const x = d.variantes[v].K[K][mod]; return `<td>${f3(x.mayoria.mIoU)} / ${f3(x.uno_a_uno.mIoU)}</td>`; }).join("") + "</tr>";
    h += "</table>"; }
  else { const d = D.eval[S.modo], K = (Object.keys(d.variantes[S.variante].K).includes(String(S.K)) ? String(S.K) : Object.keys(d.variantes[S.variante].K)[0]);
    const r = d.variantes[S.variante].K[K][S.tab === "m3" ? "M3" : S.mod];
    h += `<div class="sup">M3 = ${S.variante} · K = ${K} · evaluación: ${S.modo === "todas" ? "todas las clases" : "sin «otro»"}</div>` + resumen(r, "Agrupamiento") + tablaClases(r[S.asig].por_clase) + "<h3>Confusión</h3>" + tablaConfusion(r[S.asig]);
    if (!hayGrupos) h = "<div class='aviso'>Sin grupos por punto: solo métricas.</div>" + h; }
  el.innerHTML = h;
}
function refrescar() {
  document.getElementById("tabs").innerHTML = [["m1", "M1"], ["m2", "M2"], ["m3", "M3"], ["m4", "M4"], ["comparar", "Comparar"]].map(([k, t]) => `<button class="${S.tab === k ? "on" : ""}" onclick="irA('${k}')">${t}</button>`).join("");
  controles(); metricas(); if (S.tab !== "comparar") dibujar();
}
function irA(t) { S.tab = t; S.vista = VISTAS[t] ? VISTAS[t][0][0] : S.vista; refrescar(); }
refrescar();
</script></body></html>
"""

if __name__ == "__main__":
    main()

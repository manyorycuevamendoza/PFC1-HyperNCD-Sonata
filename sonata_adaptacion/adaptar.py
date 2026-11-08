"""Adaptacion auto-supervisada de Sonata a la Catedral (Pointcept) para Khipu.

Subcomandos (todos son CPU y rapidos; solo `entrenar` del sbatch usa GPU):
  datos      arma el dataset de Pointcept desde data/*.npy (sin PL7 ni PL13)
  parchear   aplica a Pointcept la atencion SDPA (menos memoria, mismo resultado)
  config     escribe la config de una corrida
  pesos      baja sonata.pth (necesita internet: login node) y lo adapta a Pointcept
  verificar  comprueba que las claves de los pesos coincidan con el modelo
  exportar   convierte el teacher entrenado al formato de sonata.load
"""
import argparse
import glob
import math
import os
import shutil
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(AQUI)

_VIEJO = """            attn = (q * self.scale) @ k.transpose(-2, -1)  # (N', H, K, K)
            if self.enable_rpe:
                attn = attn + self.rpe(self.get_rel_pos(point, order))
            if self.upcast_softmax:
                attn = attn.float()
            attn = self.softmax(attn)
            attn = self.attn_drop(attn).to(qkv.dtype)
            feat = (attn @ v).transpose(1, 2).reshape(-1, C)
"""


def cmd_datos(a):
    import numpy as np
    excluir = set(a.excluir)
    npys = sorted(glob.glob(os.path.join(a.data, "*.npy")))
    if not npys:
        sys.exit(f"No hay .npy en {a.data}")
    destino = os.path.join(a.salida, "train")
    marca = os.path.join(a.salida, ".completo")
    if os.path.exists(marca) and not a.rehacer:
        n = len(os.listdir(destino))
        print(f"Dataset ya armado ({n} escenas) en {a.salida}. Usa --rehacer para repetirlo.")
        return
    shutil.rmtree(a.salida, ignore_errors=True)
    usados = []
    for ruta in npys:
        nombre = os.path.basename(ruta)[:-4]
        if nombre in excluir:
            print(f"  (fuera, para evaluar) {nombre}")
            continue
        d = np.load(ruta, mmap_mode="r")
        assert d.ndim == 2 and d.shape[1] == 9, f"{nombre}: se esperaban 9 columnas, hay {d.shape}"
        out = os.path.join(destino, nombre)
        os.makedirs(out, exist_ok=True)
        np.save(f"{out}/coord.npy", np.ascontiguousarray(d[:, 0:3], dtype=np.float32))
        np.save(f"{out}/color.npy", np.ascontiguousarray(d[:, 3:6], dtype=np.float32))
        np.save(f"{out}/normal.npy", np.ascontiguousarray(d[:, 6:9], dtype=np.float32))
        cmax = float(np.max(d[:, 3:6]))
        usados.append(nombre)
        print(f"  {nombre:28s} {d.shape[0]:>11,d} puntos | color max {cmax:.0f}")
        if cmax <= 1.5:
            print("   AVISO: el color parece estar en 0-1; Sonata espera 0-255")
    open(marca, "w").write("\n".join(usados))
    print(f"\nEntrenaran {len(usados)} escenas -> {a.salida}")


def cmd_parchear(a):
    f = os.path.join(a.pointcept, "pointcept/models/point_transformer_v3/point_transformer_v3m2_sonata.py")
    src = open(f, encoding="utf-8").read()
    if "scaled_dot_product_attention" in src:
        print("Parche de atencion (SDPA) ya aplicado.")
        return
    assert _VIEJO in src, "No encontre el bloque de atencion de Pointcept (cambio el repo?)"
    orig = "".join(("    " + l) if l.strip() else l for l in _VIEJO.splitlines(True))
    nuevo = (
        "            if (not self.enable_rpe) and (not self.upcast_softmax):\n"
        "                feat = torch.nn.functional.scaled_dot_product_attention(\n"
        "                    q, k, v.to(q.dtype),\n"
        "                    dropout_p=self.attn_drop.p if self.training else 0.0,\n"
        "                    scale=self.scale,\n"
        "                )\n"
        "                feat = feat.transpose(1, 2).reshape(-1, C).to(qkv.dtype)\n"
        "            else:\n" + orig
    )
    open(f, "w", encoding="utf-8").write(src.replace(_VIEJO, nuevo))
    print("Parche de atencion (SDPA) aplicado.")


def cmd_config(a):
    base = open(f"{a.pointcept}/configs/sonata/pretrain-sonata-v1m1-0-base.py", encoding="utf-8").read()
    i, j = base.index("data = dict("), base.index("hooks = [")
    data_block = (
        "data = dict(\n"
        "    train=dict(\n"
        '        type="DefaultDataset",\n'
        '        split="train",\n'
        f'        data_root="{a.data_root}",\n'
        "        transform=transform,\n"
        "        test_mode=False,\n"
        f"        loop={a.loop},\n"
        "    )\n"
        ")\n\n"
    )
    cfg = base[:i] + data_block + base[j:]
    cambios = [
        ("batch_size = 96", f"batch_size = {a.batch}"),
        ("num_worker = 96", f"num_worker = {a.workers}"),
        # Pointcept fija loop = epoch // eval_epoch (pisa el loop del dataset).
        ("epoch = 200", f"epoch = {a.epocas * a.loop}\neval_epoch = {a.epocas}"),
        ("base_lr = 0.004", f"base_lr = {a.lr}"),
        ("enable_flash=True", "enable_flash=False"),
        ('amp_dtype = "bfloat16"', f'amp_dtype = "{a.amp}"'),
        ("save_freq=5", f"save_freq={a.save_freq}"),
        ("max_size=65536", f"max_size={a.max_puntos}"),
    ]
    for viejo, nuevo in cambios:
        assert viejo in cfg, f"No encontre '{viejo}' en la config base de Pointcept (cambio el repo?)"
        cfg = cfg.replace(viejo, nuevo)
    cfg += "\n# sin wandb (pide clave de API); el registro queda en train.log\nenable_wandb = False\n"
    os.makedirs(os.path.dirname(a.salida), exist_ok=True)
    open(a.salida, "w", encoding="utf-8").write(cfg)
    print(f"Config -> {a.salida} (epocas={a.epocas}, loop={a.loop}, batch={a.batch}, "
          f"max_puntos={a.max_puntos}, lr={a.lr}, amp={a.amp}, save_freq={a.save_freq})")


def cmd_pesos(a):
    import torch
    from huggingface_hub import hf_hub_download
    os.makedirs(a.salida, exist_ok=True)
    p = hf_hub_download("facebook/sonata", "sonata.pth")
    ck = torch.load(p, map_location="cpu", weights_only=True)
    nuevo = {}
    for k, v in ck["state_dict"].items():
        nuevo[f"student.backbone.{k}"] = v
        nuevo[f"teacher.backbone.{k}"] = v
    torch.save({"state_dict": nuevo}, f"{a.salida}/sonata_init.pth")
    torch.save(ck, f"{a.salida}/sonata_original_ckpt.pth")
    print(f"Pesos base -> {a.salida}")


def cmd_verificar(a):
    import torch
    sys.path.insert(0, a.pointcept)
    from pointcept.utils.config import Config
    from pointcept.models import build_model
    cfg = Config.fromfile(a.config)
    m = build_model(cfg.model)
    claves = set(m.state_dict().keys())
    init = torch.load(a.pesos, map_location="cpu", weights_only=False)["state_dict"]
    ok = [k for k in init if k in claves]
    mal = [k for k in init if k not in claves]
    print(f"claves iniciales: {len(init)} | coinciden: {len(ok)} | no coinciden: {len(mal)}")
    print("ejemplos que no coinciden:", mal[:5])
    sys.exit(0 if len(ok) >= 0.9 * len(init) else 3)


def cmd_exportar(a):
    import torch
    orig = torch.load(a.original, map_location="cpu", weights_only=True)
    final = os.path.join(a.run, "model", "model_last.pth")
    assert os.path.exists(final), f"No hay {final}: ¿termino al menos una epoca?"
    ck = torch.load(final, map_location="cpu", weights_only=False)["state_dict"]
    adaptado, mantenidos = {}, 0
    for k, v0 in orig["state_dict"].items():
        clave = next((p + k for p in ("teacher.backbone.", "module.teacher.backbone.") if p + k in ck), None)
        if clave is not None and ck[clave].shape == v0.shape:
            adaptado[k] = ck[clave].clone()
        else:
            adaptado[k] = v0
            mantenidos += 1
    n = len(orig["state_dict"])
    print(f"{n - mantenidos}/{n} tensores vienen del entrenamiento; {mantenidos} del Sonata original")
    assert n - mantenidos > 0.9 * n, "Casi nada coincide: revisa los nombres de claves"
    num = sum(((adaptado[k].float() - orig["state_dict"][k].float()) ** 2).sum().item() for k in adaptado)
    den = sum((orig["state_dict"][k].float() ** 2).sum().item() for k in adaptado)
    print(f"Cambio relativo de pesos: {math.sqrt(num / den):.4%}  (0 = no se entreno nada)")
    os.makedirs(os.path.dirname(a.salida), exist_ok=True)
    torch.save({"config": orig["config"], "state_dict": adaptado}, a.salida)
    print("Pesos adaptados ->", a.salida)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)

    d = sp.add_parser("datos")
    d.add_argument("--data", default=f"{REPO}/data")
    d.add_argument("--salida", default=f"{AQUI}/datos_pointcept")
    d.add_argument("--excluir", nargs="*", default=["PL7_SCN0001", "PL13_SCN0001"])
    d.add_argument("--rehacer", action="store_true")
    d.set_defaults(f=cmd_datos)

    p = sp.add_parser("parchear")
    p.add_argument("--pointcept", required=True)
    p.set_defaults(f=cmd_parchear)

    c = sp.add_parser("config")
    c.add_argument("--pointcept", required=True)
    c.add_argument("--salida", required=True)
    c.add_argument("--data-root", dest="data_root", default=f"{AQUI}/datos_pointcept")
    c.add_argument("--epocas", type=int, default=30)
    c.add_argument("--loop", type=int, default=5)
    c.add_argument("--batch", type=int, default=4)
    c.add_argument("--max-puntos", dest="max_puntos", type=int, default=65536)
    c.add_argument("--lr", type=float, default=0.0002)
    c.add_argument("--workers", type=int, default=6)
    c.add_argument("--save-freq", dest="save_freq", type=int, default=10)
    c.add_argument("--amp", default="bfloat16", choices=["bfloat16", "float16"])
    c.set_defaults(f=cmd_config)

    w = sp.add_parser("pesos")
    w.add_argument("--salida", default=f"{AQUI}/pesos_base")
    w.set_defaults(f=cmd_pesos)

    v = sp.add_parser("verificar")
    v.add_argument("--pointcept", required=True)
    v.add_argument("--config", required=True)
    v.add_argument("--pesos", required=True)
    v.set_defaults(f=cmd_verificar)

    e = sp.add_parser("exportar")
    e.add_argument("--run", required=True)
    e.add_argument("--original", default=f"{AQUI}/pesos_base/sonata_original_ckpt.pth")
    e.add_argument("--salida", required=True)
    e.set_defaults(f=cmd_exportar)

    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""foto — CLI unica de foto local. Prompt entra, caminho de imagem sai.

Subcomandos:
  foto editar   FOTO "instrucao"          edita a foto (rosto/fundo preservados)
  foto cena     "descricao" --ref A --ref B   compoe cena nova a partir de referencias
  foto vestir   FOTO ROUPA                troca a roupa da foto pela da referencia
  foto ampliar  IMAGEM [--escala 2]       upscale
  foto refs     FOTO...                   prepara referencias de identidade
  foto rosto    REAL GERADA               recoloca o rosto real na imagem gerada

Regras aprendidas na marra (nao mexa sem medir):
  * TURBO 4 steps CFG 1 e a configuracao certa neste M5. Medido: 10 steps com
    CFG 2 custou 13x mais (1767 s contra 132 s) e NAO ficou melhor.
  * Referencia de identidade tem que ser crop de ROSTO grande e bem orientado.
    Rosto pequeno ou foto girada = o modelo ignora a referencia. Use 'foto refs'.
  * Para preservar identidade de verdade, use 'editar'/'vestir' (a foto real e a
    base). 'cena' RECRIA a pessoa: sai parecida, nao identica.
"""
import argparse, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.environ.get("FOTO_PYTHON") or os.path.expanduser(
    os.environ.get("COMFYUI_DIR", "~/comfyui") + "/.venv/bin/python")


def run(script, args):
    return subprocess.call([PY, os.path.join(HERE, script)] + args)


def main():
    ap = argparse.ArgumentParser(prog="foto", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("editar", help="edita a foto (pipeline completo de 5 estagios)")
    e.add_argument("foto"); e.add_argument("instrucao")
    e.add_argument("--saida", default=None, help="arquivo final (padrao: <foto>_editada.png)")
    e.add_argument("--nao", default="", help="o que evitar")
    e.add_argument("--seed", type=int)
    e.add_argument("--mp", type=float, default=1.5)
    e.add_argument("--sem-polir", action="store_true")
    e.add_argument("--sem-cabeca", action="store_true",
                   help="use quando a pose ou o enquadramento mudarem")
    e.add_argument("--sem-grao", action="store_true")
    e.add_argument("--ampliar", action="store_true", help="SeedVR2 2x no fim")

    c = sub.add_parser("cena", help="compoe uma cena nova a partir de referencias")
    c.add_argument("descricao")
    c.add_argument("--ref", action="append", required=True)
    c.add_argument("--nao", default="cap, hat, tote bag, extra limbs, deformed hands, blurry, cartoon, plastic skin")
    c.add_argument("--tamanho", default="896x1216")
    c.add_argument("--seed", type=int); c.add_argument("--out", default="cena")

    v = sub.add_parser("vestir", help="troca a roupa da foto (descreva a roupa em texto)")
    v.add_argument("foto")
    v.add_argument("roupa", help="DESCRICAO em texto da roupa, ex: "
                   "'an oversized brown plaid flannel shirt over a white t-shirt "
                   "and wide-leg beige trousers'")
    v.add_argument("--ref-roupa", default=None,
                   help="EXPERIMENTAL: imagem da peca. So use foto da PECA ISOLADA "
                        "(catalogo, fundo liso). Foto de outra pessoa vestindo faz o "
                        "modelo colar partes dela na cena — medido e reprovado.")
    v.add_argument("--seed", type=int); v.add_argument("--out", default="vestir")

    u = sub.add_parser("ampliar", help="upscale (SeedVR2 por padrao)")
    u.add_argument("imagem"); u.add_argument("--escala", type=float, default=2.0)
    u.add_argument("--out", default=None, help="arquivo de saida (padrao: <nome>_2x.png)")
    u.add_argument("--softness", type=float, default=0.5)
    u.add_argument("--esrgan", action="store_true", help="usa ESRGAN em vez do SeedVR2")

    r = sub.add_parser("refs", help="prepara fotos de referencia de identidade")
    r.add_argument("fotos", nargs="+")

    f = sub.add_parser("rosto", help="recoloca o rosto real na imagem gerada")
    f.add_argument("real"); f.add_argument("gerada"); f.add_argument("saida")

    a = ap.parse_args()

    if a.cmd == "editar":
        saida = a.saida or (os.path.splitext(a.foto)[0] + "_editada.png")
        args = [a.foto, a.instrucao, "--saida", saida, "--mp", str(a.mp)]
        if a.nao:
            args += ["--nao", a.nao]
        if a.seed:
            args += ["--seed", str(a.seed)]
        for flag in ("sem_polir", "sem_cabeca", "sem_grao", "ampliar"):
            if getattr(a, flag):
                args.append("--" + flag.replace("_", "-"))
        return run("pipeline.py", args)

    if a.cmd == "cena":
        args = ["--prompt", a.descricao, "--out", a.out, "--size", a.tamanho,
                "--negative", a.nao]
        for ref in a.ref:
            args += ["--img", ref]
        if a.seed:
            args += ["--seed", str(a.seed)]
        return run("mage.py", args)

    if a.cmd == "vestir":
        prompt = (f"Replace the clothing worn by the person in this photograph "
                  f"with {a.roupa}. Keep the person's face, hair, body, pose and "
                  f"the entire background exactly unchanged. Do not add hats, "
                  f"bags or accessories, and do not add any other person. "
                  f"Photorealistic, natural fabric folds and lighting consistent "
                  f"with the original photo.")
        args = ["--img", a.foto, "--prompt", prompt,
                "--negative", "cap, hat, tote bag, accessory, changed face, "
                              "second person, floating head, floating shoes",
                "--out", a.out, "--mp", "1.5"]
        if a.ref_roupa:
            args = args[:2] + ["--img", a.ref_roupa] + args[2:]
        if a.seed:
            args += ["--seed", str(a.seed)]
        return run("mage.py", args)

    if a.cmd == "ampliar":
        out = a.out or os.path.splitext(a.imagem)[0] + f"_{int(a.escala)}x.png"
        args = [a.imagem, "--out", out, "--escala", str(a.escala),
                "--softness", str(a.softness)]
        if a.esrgan:
            args.append("--esrgan")
        return run("ampliar.py", args)

    if a.cmd == "refs":
        return run("prepref.py", a.fotos)

    if a.cmd == "rosto":
        return run("face1to1.py", [a.real, a.gerada, a.saida])


if __name__ == "__main__":
    sys.exit(main())

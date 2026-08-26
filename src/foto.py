#!/usr/bin/env python3
"""foto — CLI unica de foto local. Prompt entra, caminho de imagem sai.

Subcomandos:
  foto editar   FOTO "instrucao"          edita a foto (rosto/fundo preservados)
  foto cena     "descricao" --ref A --ref B   compoe cena nova a partir de referencias
  foto vestir   FOTO ROUPA                troca a roupa da foto pela da referencia
  foto gerar    "descricao" [--estilo X]  cria imagem do zero (roteamento automatico)
  foto ampliar  IMAGEM [--escala 2]       upscale
  foto refs     FOTO...                   prepara referencias de identidade

Regras aprendidas na marra (nao mexa sem medir):
  * TURBO 4 steps CFG 1 e a configuracao certa neste M5. Medido: 10 steps com
    CFG 2 custou 13x mais (1767 s contra 132 s) e NAO ficou melhor.
  * Referencia de identidade tem que ser crop de ROSTO grande e bem orientado.
    Rosto pequeno ou foto girada = o modelo ignora a referencia. Use 'foto refs'.
  * Para preservar identidade pixel a pixel, use 'editar'/'vestir' (a foto real
    e a base). 'cena' usa FLUX.2 multi-reference e recria a pessoa: confira o
    rosto antes de uso sensivel.
"""
import argparse, os, subprocess, sys

HERE = os.path.dirname(os.path.realpath(__file__))
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
    e.add_argument("--ampliar", action="store_true",
                   help="SeedVR2 2x antes de recolar a cabeca original")

    c = sub.add_parser("cena", help="compoe uma cena nova a partir de referencias")
    c.add_argument("descricao")
    c.add_argument("--ref", action="append", required=True)
    c.add_argument("--tamanho", default="896x1216")
    c.add_argument("--seed", type=int)
    c.add_argument("--saida", "--out", dest="saida", default=None)

    v = sub.add_parser("vestir", help="troca a roupa da foto (descreva a roupa em texto)")
    v.add_argument("foto")
    v.add_argument("roupa", help="DESCRICAO em texto da roupa, ex: "
                   "'an oversized brown plaid flannel shirt over a white t-shirt "
                   "and wide-leg beige trousers'")
    v.add_argument("--ref-roupa", default=None,
                   help="nao suportado neste modo; para referencias use 'foto cena'")
    v.add_argument("--seed", type=int)
    v.add_argument("--saida", "--out", dest="saida", default=None)

    g = sub.add_parser("gerar", help="cria imagem do zero, escolhendo o melhor motor")
    g.add_argument("prompt")
    g.add_argument("--saida", default=None)
    g.add_argument("--tamanho", default="1024x1024")
    g.add_argument("--seed", type=int)
    g.add_argument("--estilo", default="auto", choices=(
        "auto", "foto-natural", "iphone", "profissional", "produto",
        "cartoon", "pixel-art", "ilustracao", "anime", "famegrid", "livre"))
    g.add_argument("--motor", default="auto",
                   choices=("auto", "drawthings", "krea2", "sdxl", "flux2"))
    g.add_argument("--lora", action="append", default=[],
                   help="LoRA SDXL arquivo.safetensors[:forca]; seleciona SDXL")

    u = sub.add_parser("ampliar", help="upscale (SeedVR2 por padrao)")
    u.add_argument("imagem"); u.add_argument("--escala", type=float, default=2.0)
    u.add_argument("--out", default=None, help="arquivo de saida (padrao: <nome>_2x.png)")
    u.add_argument("--softness", type=float, default=0.5)

    r = sub.add_parser("refs", help="prepara fotos de referencia de identidade")
    r.add_argument("fotos", nargs="+")

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
        args = [a.descricao, "--tamanho", a.tamanho]
        for ref in a.ref:
            args += ["--ref", ref]
        if a.saida:
            args += ["--saida", a.saida]
        if a.seed:
            args += ["--seed", str(a.seed)]
        return run("flux2.py", args)

    if a.cmd == "vestir":
        if a.ref_roupa:
            print("erro: --ref-roupa nao e aceito por 'foto vestir': o editor pode "
                  "colar partes da pessoa da referencia. Para uma composicao nova, "
                  "use 'foto cena --ref PESSOA --ref ROUPA'.", file=sys.stderr)
            return 2
        prompt = (f"Replace the clothing worn by the person in this photograph "
                  f"with {a.roupa}. Keep the person's face, hair, body, pose and "
                  f"the entire background exactly unchanged. Do not add hats, "
                  f"bags or accessories, and do not add any other person. "
                  f"Photorealistic, natural fabric folds and lighting consistent "
                  f"with the original photo.")
        saida = a.saida or os.path.splitext(a.foto)[0] + "_vestido.png"
        args = [a.foto, prompt, "--saida", saida, "--mp", "1.5",
                "--nao", "cap, hat, tote bag, accessory, changed face, "
                         "second person, floating head, floating shoes"]
        if a.seed:
            args += ["--seed", str(a.seed)]
        return run("pipeline.py", args)

    if a.cmd == "gerar":
        args = [a.prompt, "--tamanho", a.tamanho, "--estilo", a.estilo,
                "--motor", a.motor]
        if a.saida:
            args += ["--saida", a.saida]
        if a.seed:
            args += ["--seed", str(a.seed)]
        for l in a.lora:
            args += ["--lora", l]
        return run("gerar_coringa.py", args)

    if a.cmd == "ampliar":
        out = a.out or os.path.splitext(a.imagem)[0] + f"_{int(a.escala)}x.png"
        args = [a.imagem, "--out", out, "--escala", str(a.escala),
                "--softness", str(a.softness)]
        return run("ampliar.py", args)

    if a.cmd == "refs":
        return run("prepref.py", a.fotos)

if __name__ == "__main__":
    sys.exit(main())

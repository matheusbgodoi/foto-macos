#!/usr/bin/env python3
"""Pipeline completo de edicao de foto — encadeia os 5 estagios do fluxo vencedor.

    1. EDITAR    Mage-Flow-Edit-Turbo, 4 steps, CFG 1     (~45-160 s)
    2. POLIR     SDXL denoise 0.03 + 1x-ITF-SkinDiffDetail (~130 s)
    3. AMPLIAR   SeedVR2 2x, somente na imagem editada      (~26 s, opcional)
    4. CABECA    recola a cabeca da foto original          (~1 s)
    5. GRAO      injeta o grao que falta na area editada   (instantaneo)

A ORDEM IMPORTA e foi aprendida errando. Tudo que e GENERATIVO tem de acontecer
ANTES do composite da cabeca, porque qualquer passe generativo posterior refaz o
rosto que o composite acabou de preservar. Isso vale para o POLIR e vale para o
AMPLIAR — o SeedVR2 reconstroi tudo que enxerga.
Quando se amplia, a FOTO original nunca passa pelo SeedVR2. Ela chega ao
composite por Lanczos, que muda a grade sem inventar poros, cabelo ou tracos.
O composite continua sendo o ultimo estagio que pode tocar na cabeca.
CABECA so vale quando a pose nao muda (edicao sobre a propria foto).
"""
import argparse, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.environ.get("FOTO_PYTHON") or os.path.expanduser(
    os.environ.get("COMFYUI_DIR", "~/comfyui") + "/.venv/bin/python")
OUT = os.environ.get("FOTO_OUT") or os.path.join(os.path.dirname(HERE), "out")


def step(script, args, titulo):
    print(f"\n── {titulo} " + "─" * max(0, 50 - len(titulo)))
    t0 = time.time()
    r = subprocess.run([PY, os.path.join(HERE, script)] + [str(a) for a in args],
                       capture_output=True, text=True)
    saida = (r.stdout or "") + (r.stderr or "")
    for linha in saida.splitlines():
        if linha.startswith("[") or "Traceback" in linha or "Error" in linha:
            print("   " + linha)
    if r.returncode != 0:
        print(f"   FALHOU ({time.time()-t0:.0f}s)")
        return False
    print(f"   ok ({time.time()-t0:.0f}s)")
    return True


def achar(prefixo):
    """Acha o arquivo mais recente que o ComfyUI salvou com este prefixo."""
    cands = [f for f in os.listdir(OUT) if f.startswith(prefixo) and f.endswith(".png")]
    if not cands:
        return None
    return os.path.join(OUT, max(cands, key=lambda f: os.path.getmtime(os.path.join(OUT, f))))


def imagem_valida(path):
    """Recusa quadros vazios/pretos antes de encadear modelos caros.

    O ComfyUI pode concluir o grafo e salvar um PNG preto quando um modelo fica
    em estado invalido sob pressao de memoria. Existencia do arquivo nao basta.
    """
    try:
        from PIL import Image, ImageStat
        with Image.open(path) as image:
            stats = ImageStat.Stat(image.convert("RGB").resize((64, 64)))
        luminance = sum(stats.mean) / 3
        variation = sum(stats.stddev) / 3
        return luminance >= 3.0 and variation >= 1.0
    except (OSError, ValueError):
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("foto")
    ap.add_argument("instrucao")
    ap.add_argument("--saida", required=True, help="caminho do arquivo final")
    ap.add_argument("--nao", default="", help="o que evitar (negativo)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--mp", type=float, default=1.5)
    ap.add_argument("--sem-polir", action="store_true")
    ap.add_argument("--sem-cabeca", action="store_true",
                    help="use quando a pose/enquadramento mudarem")
    ap.add_argument("--sem-grao", action="store_true")
    ap.add_argument("--ampliar", action="store_true",
                    help="SeedVR2 2x antes do composite final da cabeca")
    a = ap.parse_args()

    foto = os.path.abspath(os.path.expanduser(a.foto))
    saida = os.path.abspath(os.path.expanduser(a.saida))
    os.makedirs(OUT, exist_ok=True)
    tag = f"pl{int(time.time()) % 100000}"
    t0 = time.time()

    # 1. EDITAR
    args = ["--img", foto, "--prompt", a.instrucao, "--out", tag + "e", "--mp", a.mp]
    if a.nao:
        args += ["--negative", a.nao]
    if a.seed:
        args += ["--seed", a.seed]
    if not step("mage.py", args, "1/5 editar (Mage-Flow-Edit-Turbo)"):
        sys.exit(1)
    atual = achar(tag + "e")
    if not atual or not imagem_valida(atual):
        print("erro: a edicao produziu arquivo ausente, preto ou sem variacao; "
              "pipeline interrompido antes do polimento")
        sys.exit(1)

    # 2. POLIR
    if not a.sem_polir:
        if step("polir.py", [atual, "--out", tag + "p", "--escala", 1.0],
                "2/5 polir (SDXL denoise 0.03 + pele)"):
            polido = achar(tag + "p")
            if polido and imagem_valida(polido):
                atual = polido
            else:
                print("   polimento invalido/preto; mantendo a edicao valida")

    # 3. AMPLIAR — ANTES de recolar a cabeca. O SeedVR2 e generativo: se rodar
    #    depois, ele re-gera justamente o rosto que o composite preservou.
    #    Ele toca SO na editada. A foto original entra no composite por Lanczos;
    #    passar a original pelo SeedVR2 seria regenerar o rosto por outra porta.
    if a.ampliar:
        alvo = os.path.join(OUT, tag + "_2x.png")
        if step("ampliar.py", [atual, "--out", alvo], "3/5 ampliar editada (SeedVR2 2x)"):
            atual = alvo

    # 4. CABECA
    if not a.sem_cabeca:
        alvo = os.path.join(OUT, tag + "_head.png")
        cabeca_args = [foto, atual, alvo]
        if a.ampliar:
            # Mantem a grade ampliada, mas desliga a sintese de nitidez dentro
            # da cabeca: identidade estrita vale mais que detalhe inventado.
            cabeca_args += ["--grade", "editada", "--nitidez", "nao"]
        if step("cabeca.py", cabeca_args, "4/5 recolar cabeca original"):
            atual = alvo

    # 5. GRAO
    if not a.sem_grao:
        alvo = os.path.join(OUT, tag + "_grao.png")
        if step("grao.py", [foto, atual, alvo], "5/5 casar grao"):
            atual = alvo

    os.makedirs(os.path.dirname(saida) or ".", exist_ok=True)
    import shutil
    shutil.copy(atual, saida)
    from PIL import Image
    print(f"\n✓ {saida}  {Image.open(saida).size}  — {time.time()-t0:.0f}s no total")


if __name__ == "__main__":
    main()

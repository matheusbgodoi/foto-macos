#!/usr/bin/env python3
"""Servidor MCP do foto-macos — edicao e geracao de imagem local.

Este e o unico conector publico. Ele roteia geracao e edicao entre ComfyUI,
Draw Things e MLX; o usuario nao precisa conhecer o backend.

Registrar no Claude Code:
    claude mcp add --scope user foto-macos -- ~/comfyui/.venv/bin/python \\
        ~/src/foto-macos/src/mcp_server.py

Registrar no CRIAs AI / Codex: novo conector "stdio" com o mesmo comando.

Precisa do ComfyUI no ar em 127.0.0.1:8188 — use a ferramenta foto_status.
"""
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.environ.get("FOTO_PYTHON") or os.path.expanduser(
    os.environ.get("COMFYUI_DIR", "~/comfyui") + "/.venv/bin/python")
COMFY = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")

try:
    from mcp.server.mcpserver import MCPServer
except ImportError:
    sys.exit("falta o pacote mcp (2.x): uv pip install 'mcp>=2'")

app = MCPServer(
    name="foto-macos",
    instructions=(
        "Edicao de foto local por instrucao, preservando o rosto real da pessoa. "
        "foto_editar altera uma foto QUE JA EXISTE (trocar roupa, remover objeto, "
        "trocar fundo) preservando o rosto real. foto_gerar cria imagem do zero "
        "e escolhe automaticamente o motor Apple mais adequado. foto_cena usa "
        "uma ou mais imagens de referencia para criar uma composicao nova."
    ),
)


def _comfy_ok() -> bool:
    try:
        urllib.request.urlopen(COMFY + "/system_stats", timeout=3)
        return True
    except Exception:
        return False


def _rodar(script: str, args: list, timeout: int = 1800):
    r = subprocess.run([PY, os.path.join(HERE, script)] + [str(x) for x in args],
                       capture_output=True, text=True, timeout=timeout)
    return (r.stdout or "") + (r.stderr or ""), r.returncode


def _exige_comfy():
    if _comfy_ok():
        return None
    return (f"ComfyUI nao responde em {COMFY}. Suba com:\n"
            f"  cd ~/comfyui && ./.venv/bin/python main.py "
            f"--use-pytorch-cross-attention --reserve-vram 2 --listen 127.0.0.1")


@app.tool(
    description=(
        "Edita uma foto existente por instrucao, preservando o rosto REAL da pessoa. "
        "Serve para trocar roupa, remover ou trocar objetos e mudar o fundo. "
        "Leva ~3 min e devolve o caminho do arquivo gerado. "
        "Escreva a instrucao em ingles e diga tambem o que PRESERVAR."
    )
)
def foto_editar(
    foto: str,
    instrucao: str,
    saida: str = "",
    evitar: str = "",
    ampliar: bool = False,
    sem_cabeca: bool = False,
    seed: int = 0,
) -> str:
    """Edita a foto e devolve o caminho do resultado.

    Args:
        foto: caminho absoluto da foto a editar.
        instrucao: o que mudar. Ex: "Replace the shirt with a black hoodie.
            Keep his face, hair, hands and the background unchanged."
        saida: caminho do arquivo final. Vazio = <foto>_editada.png.
        evitar: prompt negativo, ex: "logo, text, changed face".
        ampliar: aplica SeedVR2 2x antes de recolar a cabeca (+26 s).
        sem_cabeca: desliga a preservacao do rosto. Use so se a pose ou o
            enquadramento mudarem — nesses casos recolar a cabeca nao faz sentido.
        seed: 0 = aleatoria.
    """
    erro = _exige_comfy()
    if erro:
        return erro
    caminho = os.path.abspath(os.path.expanduser(foto))
    if not os.path.exists(caminho):
        return f"nao encontrei o arquivo: {caminho}"
    destino = os.path.abspath(os.path.expanduser(
        saida or os.path.splitext(caminho)[0] + "_editada.png"))

    args = [caminho, instrucao, "--saida", destino]
    if evitar:
        args += ["--nao", evitar]
    if seed:
        args += ["--seed", seed]
    if ampliar:
        args.append("--ampliar")
    if sem_cabeca:
        args.append("--sem-cabeca")

    saida_txt, rc = _rodar("pipeline.py", args)
    if rc != 0 or not os.path.exists(destino):
        return f"falhou:\n{saida_txt[-1500:]}"
    return f"{destino}\n\n{saida_txt[-600:]}"


@app.tool(description="Amplia uma imagem com SeedVR2, reconstruindo microtextura real. ~26 s para 2x.")
def foto_ampliar(imagem: str, saida: str = "", escala: float = 2.0) -> str:
    """Amplia e devolve o caminho do resultado."""
    img = os.path.abspath(os.path.expanduser(imagem))
    destino = os.path.abspath(os.path.expanduser(
        saida or os.path.splitext(img)[0] + "_2x.png"))
    txt, rc = _rodar("ampliar.py", [img, "--out", destino, "--escala", escala])
    return destino if (rc == 0 and os.path.exists(destino)) else f"falhou:\n{txt[-1000:]}"


@app.tool(
    description=(
        "Prepara fotos para servirem de referencia de identidade: corrige a orientacao "
        "testando as 4 rotacoes e recorta o rosto em 1024 px. Rode antes de usar qualquer "
        "foto como referencia — foto girada ou rosto pequeno faz o modelo descartar a "
        "referencia em silencio."
    )
)
def foto_referencias(fotos: list[str]) -> str:
    """Prepara as referencias e devolve o relatorio por arquivo."""
    txt, _ = _rodar("prepref.py", [os.path.expanduser(f) for f in fotos], timeout=300)
    return txt[-1500:]


@app.tool(
    description=(
        "Cria uma imagem DO ZERO e escolhe o motor automaticamente: Draw Things + "
        "Z-Image para fotografia/estilos rapidos; ComfyUI + SDXL quando uma LoRA SDXL "
        "for fornecida; FLUX.2 quando explicitamente pedido. Para editar uma foto "
        "existente use foto_editar; para varias referencias use foto_cena."
    )
)
def foto_gerar(
    prompt: str,
    saida: str = "",
    tamanho: str = "1024x1024",
    seed: int = 0,
    estilo: str = "auto",
    motor: str = "auto",
    loras: list[str] | None = None,
) -> str:
    """Gera a imagem e devolve o caminho.

    Args:
        prompt: descricao da imagem; portugues e aceito.
        saida: caminho do arquivo final.
        tamanho: "LARGURAxALTURA"; perto de 1 MP e o melhor custo/qualidade.
        seed: 0 = aleatoria.
        estilo: auto, foto-natural, iphone, profissional, produto, cartoon,
            pixel-art, ilustracao, anime ou livre.
        motor: auto, drawthings, sdxl ou flux2. Deixe auto normalmente.
        loras: LoRAs SDXL em models/loras, opcionalmente "nome:forca". Quando
            presentes, o roteador seleciona SDXL.
    """
    destino = os.path.abspath(os.path.expanduser(saida or os.path.join(
        "~/Downloads", f"foto_{int(time.time())}.png")))
    args = [prompt, "--tamanho", tamanho, "--estilo", estilo,
            "--motor", motor, "--saida", destino]
    if seed:
        args += ["--seed", seed]
    for l in (loras or []):
        args += ["--lora", l]
    txt, rc = _rodar("gerar_coringa.py", args)
    if rc != 0 or not os.path.exists(destino):
        return f"falhou:\n{txt[-1200:]}"
    return f"{destino}\n\n{txt[-600:]}"


@app.tool(
    description=(
        "Cria uma CENA NOVA usando 1-4 imagens como referencias de pessoa, roupa, "
        "objeto ou estilo. Usa FLUX.2 Klein 4B multi-reference. Diferente de "
        "foto_editar: a composicao e a pose podem mudar. A identidade e zero-shot; "
        "confira o rosto antes de uso sensivel."
    )
)
def foto_cena(
    prompt: str,
    referencias: list[str],
    saida: str = "",
    tamanho: str = "896x1216",
    seed: int = 0,
) -> str:
    """Gera uma composicao nova a partir de varias referencias."""
    erro = _exige_comfy()
    if erro:
        return erro
    if not referencias:
        return "forneca pelo menos uma imagem em referencias"
    refs = [os.path.abspath(os.path.expanduser(path)) for path in referencias[:4]]
    faltando = [path for path in refs if not os.path.exists(path)]
    if faltando:
        return "nao encontrei: " + ", ".join(faltando)
    destino = os.path.abspath(os.path.expanduser(saida or os.path.join(
        "~/Downloads", f"cena_{int(time.time())}.png")))
    args = [prompt, "--tamanho", tamanho, "--saida", destino]
    for path in refs:
        args += ["--ref", path]
    if seed:
        args += ["--seed", seed]
    txt, rc = _rodar("flux2.py", args)
    if rc != 0 or not os.path.exists(destino):
        return f"falhou:\n{txt[-1400:]}"
    return f"{destino}\n\n{txt[-600:]}"


@app.tool(description="Verifica se o ComfyUI esta no ar e quais modelos do pipeline estao instalados.")
def foto_status() -> str:
    """Diagnostico do ambiente."""
    m = os.path.expanduser(os.environ.get("COMFYUI_DIR", "~/comfyui") + "/models")
    precisa = {
        "editar": f"{m}/diffusion_models/mage_flow_edit_turbo_bf16.safetensors",
        "encoder": f"{m}/text_encoders/qwen3vl_4b_bf16.safetensors",
        "vae": f"{m}/vae/mage_flow_vae_bf16.safetensors",
        "polir": f"{m}/checkpoints/RealVisXL_V5.0_fp16.safetensors",
        "pele": f"{m}/upscale_models/1x-ITF-SkinDiffDetail-Lite-v1.pth",
        "gerar/referencias": f"{m}/diffusion_models/flux-2-klein-4b.safetensors",
    }
    linhas = [f"ComfyUI em {COMFY}: {'no ar' if _comfy_ok() else 'FORA DO AR'}"]
    for k, v in precisa.items():
        linhas.append(f"  {'ok   ' if os.path.exists(v) else 'FALTA'} {k}: {os.path.basename(v)}")
    draw_cli = os.path.expanduser(os.environ.get(
        "DRAWTHINGS_BIN", "/opt/homebrew/bin/draw-things-cli"))
    draw_model = os.path.expanduser(os.path.join(os.environ.get(
        "DRAWTHINGS_MODELS_DIR",
        "~/Library/Application Support/local-photo-ai-m5/models"),
        "z_image_turbo_1.0_i8x.ckpt"))
    draw_ok = os.path.isfile(draw_cli) and os.path.isfile(draw_model)
    linhas.append(f"  {'ok   ' if draw_ok else 'opcional ausente'} gerar rapido: Draw Things + Z-Image i8x")
    return "\n".join(linhas)


if __name__ == "__main__":
    app.run(transport="stdio")

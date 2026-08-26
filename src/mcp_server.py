#!/usr/bin/env python3
"""Servidor MCP do foto-macos — edicao e geracao de imagem local.

Sobrepoe em parte o MCP `local-photo` (Draw Things + Z-Image), que so gera do
zero. Este aqui gera E edita, e na edicao preserva o rosto real da pessoa.

Registrar no Claude Code:
    claude mcp add foto-edit -- ~/comfyui/.venv/bin/python \\
        ~/src/foto-macos/src/mcp_server.py

Registrar no CRIAs AI / Codex: novo conector "stdio" com o mesmo comando.

Precisa do ComfyUI no ar em 127.0.0.1:8188 — use a ferramenta foto_status.
"""
import os
import subprocess
import sys
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
        "trocar fundo) preservando o rosto real. foto_gerar cria imagem do zero a "
        "partir de texto, com SDXL e LoRAs de estilo. "
        "Limite conhecido: nao coloca uma pessoa numa cena nova a partir de uma "
        "selfie — o rosto sairia parecido, nao identico."
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
        ampliar: aplica SeedVR2 2x no fim (+26 s).
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
    erro = _exige_comfy()
    if erro:
        return erro
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
        "Cria uma imagem DO ZERO a partir de texto, com SDXL. Aceita LoRAs de estilo "
        "(desenho, anime, fotografia, produto) — o ecossistema SDXL tem dezenas de "
        "milhares delas. ~50 s a 1 MP. Para EDITAR uma foto existente use foto_editar."
    )
)
def foto_gerar(
    prompt: str,
    saida: str = "",
    tamanho: str = "1024x1024",
    steps: int = 25,
    cfg: float = 5.0,
    seed: int = 0,
    loras: list[str] | None = None,
) -> str:
    """Gera a imagem e devolve o caminho.

    Args:
        prompt: descricao da cena, em ingles funciona melhor.
        saida: caminho do arquivo final.
        tamanho: "LARGURAxALTURA". SDXL rende melhor perto de 1 MP.
        steps: 25 e um bom padrao; abaixo de 15 perde detalhe.
        cfg: 5.0. Acima de 8 satura e endurece a pele.
        seed: 0 = aleatoria.
        loras: nomes de arquivo em models/loras, opcionalmente "nome:forca".
    """
    erro = _exige_comfy()
    if erro:
        return erro
    args = [prompt, "--tamanho", tamanho, "--steps", steps, "--cfg", cfg]
    if saida:
        args += ["--saida", os.path.abspath(os.path.expanduser(saida))]
    if seed:
        args += ["--seed", seed]
    for l in (loras or []):
        args += ["--lora", l]
    txt, rc = _rodar("gerar.py", args)
    if rc != 0:
        return f"falhou:\n{txt[-1200:]}"
    return txt.strip().splitlines()[-1]


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
    }
    linhas = [f"ComfyUI em {COMFY}: {'no ar' if _comfy_ok() else 'FORA DO AR'}"]
    for k, v in precisa.items():
        linhas.append(f"  {'ok   ' if os.path.exists(v) else 'FALTA'} {k}: {os.path.basename(v)}")
    return "\n".join(linhas)


if __name__ == "__main__":
    app.run(transport="stdio")

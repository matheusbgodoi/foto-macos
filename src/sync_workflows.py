#!/usr/bin/env python3
"""Instala no ComfyUI os workflows visuais que correspondem aos motores locais.

Os arquivos-base sao os templates oficiais empacotados com a propria versao do
ComfyUI. Este script troca apenas os nomes/variantes pelos pesos instalados no
Mac e grava a mesma copia versionada no repositorio e em user/default/workflows.
"""
import json
import os
import shutil
import sys
import site
from pathlib import Path

COMFY = Path(os.path.expanduser(os.environ.get("COMFYUI_DIR", "~/comfyui")))
ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_CANDIDATES = list(COMFY.glob(
    ".venv/lib/python*/site-packages/comfyui_workflow_templates_json/templates")) + [
    Path(folder) / "comfyui_workflow_templates_json/templates"
    for folder in site.getsitepackages()
]
TEMPLATES = next((path for path in TEMPLATE_CANDIDATES if path.exists()),
                 TEMPLATE_CANDIDATES[0])
REPO_OUT = ROOT / "workflows/comfyui"
USER_OUT = COMFY / "user/default/workflows/foto-macos"


def replace_widgets(data, node_type, replacements):
    scopes = [data]
    scopes += data.get("definitions", {}).get("subgraphs", [])
    for scope in scopes:
        for node in scope.get("nodes", []):
            if node.get("type") != node_type:
                continue
            values = node.get("widgets_values")
            if not isinstance(values, list):
                continue
            for index, value in replacements.items():
                if index < len(values):
                    values[index] = value


def patch_strings(value, replacements):
    if isinstance(value, dict):
        return {key: patch_strings(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [patch_strings(item, replacements) for item in value]
    if isinstance(value, str):
        for old, new in replacements.items():
            value = value.replace(old, new)
    return value


def install(source_name, output_name, replacements=None, node_patches=None):
    source = TEMPLATES / source_name
    if not source.exists():
        raise FileNotFoundError(f"template oficial nao encontrado: {source}")
    data = json.loads(source.read_text())
    if replacements:
        data = patch_strings(data, replacements)
    for node_type, values in (node_patches or {}).items():
        replace_widgets(data, node_type, values)
    data.setdefault("extra", {})["foto_macos"] = {
        "source": source_name,
        "generated_by": "src/sync_workflows.py",
        "warning": "arquivo gerado; edite uma copia se quiser preservar mudancas manuais",
    }
    for directory in (REPO_OUT, USER_OUT):
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / output_name
        target.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        print(target)


def main():
    if not TEMPLATES.exists():
        sys.exit("templates do ComfyUI nao encontrados; instale "
                 "comfyui-workflow-templates na venv do ComfyUI")

    install(
        "image_mage_flow_edit_turbo_int8.json",
        "01 - Editar foto - Mage Flow Turbo.json",
        {"mage_flow_edit_turbo_int8_convrot.safetensors":
         "mage_flow_edit_turbo_bf16.safetensors"},
    )
    install(
        "image_flux2_klein_image_edit_4b_distilled.json",
        "02 - Referencias e cena - FLUX2 Klein 4B.json",
        {"flux-2-klein-4b-fp8.safetensors": "flux-2-klein-4b.safetensors"},
    )
    install(
        "image_flux2_klein_text_to_image.json",
        "03 - Gerar versatil - FLUX2 Klein 4B.json",
        {"flux-2-klein-4b-fp8.safetensors": "flux-2-klein-4b.safetensors"},
    )
    apple_fast = ROOT / "workflows/apple/04 - Gerar rapido - Z-Image Draw Things.json"
    for directory in (REPO_OUT, USER_OUT):
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copy2(apple_fast, directory / "04 - Gerar rapido - Z-Image Draw Things.json")
        obsolete = directory / "04 - Gerar rapido - Z-Image Turbo.json"
        if obsolete.exists():
            obsolete.unlink()

    # Krea 2 usa o mesmo runtime MLX do CLI/MCP. Copiar o workflow versionado
    # evita duplicar o checkpoint em formato ComfyUI somente para a interface.
    krea_workflow = REPO_OUT / "05 - Fotorrealismo - Krea 2 Famegrid MLX.json"
    USER_OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(krea_workflow, USER_OUT / krea_workflow.name)

    readme = """# Workflows visuais

Estes arquivos sao gerados por `src/sync_workflows.py` a partir dos templates
oficiais da versao instalada do ComfyUI. Eles aparecem em **Workflows > Browse**
na pasta `foto-macos` depois de recarregar `http://127.0.0.1:8188`.

1. **Editar foto / Mage** — mesmo motor do `foto editar`, antes do polimento e
   do composite externo.
2. **Referencias e cena / FLUX.2** — pessoa, roupa e estilo como referencias;
   candidato para `foto cena`.
3. **Gerar versatil / FLUX.2** — texto para imagem local no ComfyUI.
4. **Gerar rapido / Z-Image** — chama o mesmo Draw Things i8x do CLI/MCP; nao
   mantem uma copia BF16 mais lenta dentro do ComfyUI. O no visual roteia
   apenas runtimes externos Apple; SDXL/FLUX usam os grafos nativos 1--3 para
   nao chamar a fila do ComfyUI de dentro dela mesma.
5. **Fotorrealismo / Krea 2 + Famegrid** — chama o mesmo runtime MLX Q4 do
   CLI/MCP por um no customizado; nao duplica o checkpoint dentro do ComfyUI.

O pipeline completo de preservacao de identidade tambem usa Vision.framework
e SeedVR2/MLX. Essas etapas nao sao nos do ComfyUI; o grafo visual mostra o
nucleo de difusao, enquanto o CLI/MCP orquestra o acabamento externo.
"""
    for directory in (REPO_OUT, USER_OUT):
        (directory / "README.md").write_text(readme)


if __name__ == "__main__":
    main()

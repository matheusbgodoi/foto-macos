# Conectar aos agentes

Existe um único servidor público: **`foto-macos`**. Ele expõe geração, edição,
referências e upscale e roteia internamente entre ComfyUI, Draw Things e MLX.
O antigo `local-photo` pode continuar instalado como dependência interna, mas
não deve ser registrado como segundo MCP.

Comando stdio:

```text
/Users/matheusbgodoi/comfyui/.venv/bin/python /Users/matheusbgodoi/src/foto-macos/src/mcp_server.py
```

## Claude Code

```bash
claude mcp add --scope user foto-macos -- \
  ~/comfyui/.venv/bin/python ~/src/foto-macos/src/mcp_server.py
claude mcp list
```

## Codex

Em `~/.codex/config.toml`:

```toml
[mcp_servers.foto-macos]
command = "/Users/matheusbgodoi/comfyui/.venv/bin/python"
args = ["/Users/matheusbgodoi/src/foto-macos/src/mcp_server.py"]
```

## CRIAs AI / Local Studio

Crie um conector local/stdio ou acrescente ao array `connectors` de
`~/Library/Application Support/Local Studio/connectors.json`:

```json
{
  "id": "foto-macos",
  "name": "Foto MacOS — Gerar e Editar",
  "transport": "stdio",
  "command": "/Users/matheusbgodoi/comfyui/.venv/bin/python",
  "args": ["/Users/matheusbgodoi/src/foto-macos/src/mcp_server.py"],
  "env": {},
  "enabled": true
}
```

Feche o app antes de editar o arquivo e reinicie depois.

## Ferramentas

| ferramenta | uso |
|---|---|
| `foto_gerar` | imagem do zero; roteamento automático por estilo/LoRA |
| `foto_cena` | nova composição com 1–4 referências via FLUX.2 |
| `foto_editar` | altera foto existente e preserva a cabeça original |
| `foto_ampliar` | SeedVR2/MLX; Lanczos se o modelo falhar |
| `foto_referencias` | corrige orientação e cria crops de rosto |
| `foto_status` | testa ComfyUI e modelos |

O ComfyUI precisa estar ativo para editar, compor referências e usar os
backends Comfy. Geração via Draw Things e upscale via MLX são independentes.

```bash
cd ~/comfyui
./.venv/bin/python main.py --use-pytorch-cross-attention \
  --reserve-vram 2 --listen 127.0.0.1
```

Variáveis: `COMFYUI_DIR`, `COMFYUI_URL`, `FOTO_PYTHON`, `FOTO_OUT`,
`LOCAL_PHOTO_BIN`, `DRAWTHINGS_BIN` e `DRAWTHINGS_MODELS_DIR`.

# Conectar aos agentes (CRIAs AI, Claude Code, Codex)

## Quem é quem

Existem **dois** servidores locais de imagem, e eles fazem coisas diferentes.
Confundir os dois é fácil — o nome de ambos começa com "photo/foto".

| conector | faz | motor | quando usar |
|---|---|---|---|
| **`local-photo`** ("Local Photo AI") | **gera** imagem do zero, a partir de texto | Draw Things + Z-Image | "crie uma foto de um café ao entardecer" |
| **`foto-edit`** (este repositório) | **edita** uma foto que já existe, preservando o rosto real | ComfyUI + Mage-Flow-Edit | "troque a camiseta desta foto minha por um moletom preto" |

Regra prática: **não existe foto de entrada → `local-photo`. Existe foto de
entrada → `foto-edit`.**

---

## Registrar

O comando é o mesmo nos três clientes — muda só onde você cola.

```
~/comfyui/.venv/bin/python ~/src/foto-macos/src/mcp_server.py
```

### Claude Code

```bash
claude mcp add --scope user foto-edit -- \
  ~/comfyui/.venv/bin/python ~/src/foto-macos/src/mcp_server.py
```

Use `--scope user`, senão o conector só funciona quando o Claude Code está
rodando dentro da pasta do projeto.

Conferir: `claude mcp list` deve mostrar `foto-edit … ✔ Connected`.

### CRIAs AI

`Configure → Integrations → Connectors → + New connector`, tipo **stdio**:

- **Command:** `/Users/matheusbgodoi/comfyui/.venv/bin/python`
- **Args:** `/Users/matheusbgodoi/src/foto-macos/src/mcp_server.py`

### Codex

No `~/.codex/config.toml`:

```toml
[mcp_servers.foto-edit]
command = "/Users/matheusbgodoi/comfyui/.venv/bin/python"
args = ["/Users/matheusbgodoi/src/foto-macos/src/mcp_server.py"]
```

---

## Ferramentas expostas

| ferramenta | o que faz | tempo |
|---|---|---|
| `foto_editar` | edita por instrução, preservando o rosto | ~3 min |
| `foto_ampliar` | SeedVR2 2× | ~26 s |
| `foto_referencias` | corrige orientação e recorta o rosto em 1024 px | segundos |
| `foto_status` | ComfyUI está no ar? modelos instalados? | instantâneo |

O `foto_editar` roda o pipeline de cinco estágios inteiro — não é preciso
orquestrar as etapas na mão.

---

## Pré-requisito

O ComfyUI precisa estar no ar:

```bash
cd ~/comfyui && ./.venv/bin/python main.py \
  --use-pytorch-cross-attention --reserve-vram 2 --listen 127.0.0.1
```

Se não estiver, todas as ferramentas devolvem uma mensagem dizendo isso, com o
comando para subir — em vez de falharem de forma obscura. `foto_status` também
lista quais modelos estão faltando.

## Variáveis de ambiente

| variável | padrão | para quê |
|---|---|---|
| `COMFYUI_DIR` | `~/comfyui` | onde está o ComfyUI |
| `COMFYUI_URL` | `http://127.0.0.1:8188` | endereço da API |
| `FOTO_PYTHON` | `$COMFYUI_DIR/.venv/bin/python` | interpretador dos scripts |
| `FOTO_OUT` | `<repo>/out` | onde ficam os intermediários |

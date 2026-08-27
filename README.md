# foto-macos

Uma porta única para gerar e editar imagens localmente em Apple Silicon.
O CLI e o servidor MCP escolhem o backend adequado; você não precisa decidir
entre ComfyUI, Draw Things e MLX a cada pedido.

```bash
foto gerar "foto casual de iPhone, uma pessoa esperando o ônibus na chuva"
foto editar foto.jpg "troque a camiseta por um moletom preto"
foto cena "eu de lado em um museu" --ref rosto.png --ref corpo.png --ref roupa.png
```

Desenvolvido e medido num MacBook Pro M5 base com 24 GB. O projeto é local e
não envia fotos para APIs externas.

## O roteador

| pedido | backend | motivo |
|---|---|---|
| geração rápida/fotográfica | Draw Things + Z-Image Turbo i8x | runtime Metal quantizado; 54,7 s no teste “iPhone” |
| fotorrealismo máximo ou identidade cadastrada | MLX + Krea 2 Turbo Q4 + LoRA opcional | melhor estética natural; cerca de 80–110 s em 640×896 com o modelo em cache |
| geração sem Draw Things | ComfyUI + FLUX.2 Klein 4B | fallback aberto e instalado pelo projeto |
| geração com LoRA SDXL | ComfyUI + RealVisXL | ecossistema amplo de estilos/LoRAs |
| cena nova com referências | ComfyUI + FLUX.2 Klein 4B | edição multi-referência nativa |
| editar uma foto existente | Mage-Flow-Edit-Turbo + acabamento | preserva a fotografia fora da alteração |
| ampliar | SeedVR2 via MFLUX/MLX | reconstrução opcional de microtextura |

O Draw Things continua instalado como motor interno rápido, mas não aparece
como um segundo conector. O único conector que agentes veem chama-se
`foto-macos`. Quando uma edição/cena precisa do ComfyUI e ele está desligado, o
MCP inicia o serviço automaticamente; os backends externos não pagam esse custo.

## Edição com identidade estrita

Modelos de difusão reescrevem a imagem inteira, mesmo quando a instrução pede
uma alteração pequena. O fluxo de edição evita que isso destrua o rosto:

```text
foto original ─────────────────────────────────────────────┐
  1. Mage-Flow Edit Turbo (4 steps, CFG 1)                │
  2. polimento SDXL, denoise 0,03 + detalhe de pele       │
  3. SeedVR2 somente na imagem editada (opcional)         │
  4. cabeça original: Vision + tom + pirâmide Laplaciana ◄┤
  5. casamento de grão                                   ◄┘
```

A ordem é intencional: nenhum estágio generativo toca o rosto depois que os
pixels originais são recolocados. Quando há upscale, a cabeça original muda de
grade por Lanczos, não por SeedVR2. A máscara usa segmentação do
Vision.framework e blend multi-banda; não é uma elipse com feather largo.

Esse modo pressupõe que cabeça e enquadramento não se moveram. Para uma cena
nova, use `foto cena`; não tente colar a cabeça de uma pose diferente.

## Instalação

Requer macOS Apple Silicon, [ComfyUI](https://github.com/Comfy-Org/ComfyUI) em
`~/comfyui` e [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/matheusbgodoi/foto-macos.git
cd foto-macos
./install.sh
```

O instalador baixa os modelos do ComfyUI, instala o CLI em `~/.local/bin/foto`,
instala os workflows visuais e aplica a correção do SeedVR2 documentada em
[docs/BUGS.md](docs/BUGS.md). Draw Things é opcional: quando ele e seus pesos
i8x não existem, `foto gerar` cai automaticamente no FLUX.2/ComfyUI.

Modelos, tamanhos e licenças: [docs/MODELOS.md](docs/MODELOS.md).
Comparação de modelos, funções e tempos: [docs/COMPARACAO.md](docs/COMPARACAO.md).
Tutorial operacional: [docs/TUTORIAL.md](docs/TUTORIAL.md).
Treino local de identidade: [docs/IDENTIDADE.md](docs/IDENTIDADE.md).

## Uso

```bash
foto gerar "uma livraria aconchegante à noite" --estilo foto-natural
foto gerar "um mago numa torre" --estilo cartoon
foto gerar "uma cidade espacial" --estilo pixel-art --tamanho 1024x1024

foto gerar "retrato editorial" --motor flux2
foto gerar "retrato casual numa cozinha real" --estilo famegrid
foto gerar "foto comum de iPhone num bar" --motor krea2 --estilo iphone
foto gerar "Pessoa apresentando uma palestra de tecnologia" # se Pessoa estiver cadastrada
foto gerar "ilustração" --lora minha_lora.safetensors:0.8

foto editar foto.jpg "Replace the blue shirt with a black hoodie. Keep face, hair, hands and background unchanged."
foto vestir foto.jpg "a plain black cotton hoodie without logos"

foto cena "natural candid photo of this person in a museum wearing the referenced outfit" \
  --ref rosto.png --ref corpo.png --ref roupa.png

foto refs selfie1.heic selfie2.jpg
```

Estilos: `auto`, `foto-natural`, `iphone`, `profissional`, `produto`,
`cartoon`, `pixel-art`, `ilustracao`, `anime`, `famegrid` e `livre`.

## Workflows no localhost

Os grafos ficam versionados em [`workflows/comfyui`](workflows/comfyui) e são
copiados para `~/comfyui/user/default/workflows/foto-macos/`.

Recarregue `http://127.0.0.1:8188` e abra **Workflows → Browse → foto-macos**:

1. Editar foto — Mage-Flow Turbo
2. Referências e cena — FLUX.2 Klein 4B
3. Gerar versátil — FLUX.2 Klein 4B
4. Gerar rápido — Z-Image/Draw Things i8x (mesmo motor do CLI)
5. Fotorrealismo — Krea 2/Famegrid MLX (mesmo motor do CLI)
6. Identidade por referência — Krea 2 ReID (receita oficial para CUDA; não é
   executável de produção no MPS deste Mac)

Os grafos representam o núcleo de difusão. Vision.framework e SeedVR2/MLX são
processos externos ao ComfyUI e, por isso, o pipeline completo continua sendo
orquestrado pelo CLI/MCP.

## MCP para agentes

```bash
claude mcp add --scope user foto-macos -- \
  ~/comfyui/.venv/bin/python ~/src/foto-macos/src/mcp_server.py
```

Ferramentas: `foto_gerar`, `foto_cena`, `foto_editar`, `foto_ampliar`,
`foto_referencias`, `foto_status`, `civitai_modelo` e `civitai_baixar`.
Configuração para Claude Code, Codex, OpenCode, Pi e Local Studio:
[docs/CONECTORES.md](docs/CONECTORES.md).

## Resultados medidos neste M5

| operação | tempo observado |
|---|---:|
| geração “iPhone”, Z-Image/Draw Things, 768×1024 | **54,7 s** |
| Krea 2 Turbo Q4, 640×896, 8 passos | **~80–110 s** (pico observado 16,8 GB) |
| Krea 2 Turbo Q8, 640×896, 8 passos | **360 s** (pico 23,2 GB; fora da produção) |
| geração SDXL, 896×1152 | **48 s** |
| cena FLUX.2, 3 referências, 896×1216 | **212,6 s** |
| edição Mage, ~1 MP | **44–160 s** |
| polimento SDXL | **~132 s** |
| SeedVR2 2× | **25,8 s** |
| pipeline de edição completo | **~3–4 min** |

Mais passos não ajudaram neste hardware: Mage não-turbo com 10 steps/CFG 2
levou 1.767 s e perdeu qualidade. O regime distilled/turbo de 4 steps é o
padrão medido.

## Limites honestos

- `foto editar` preserva a cabeça pixel a pixel quando pose e enquadramento não
  mudam; a região gerada ainda deve ser inspecionada para acessórios ou membros
  inventados.
- `foto cena` preserva características zero-shot muito melhor com FLUX.2 e
  múltiplas referências, mas não oferece garantia biométrica/pixel-idêntica.
- SeedVR2 é generativo. Use upscale apenas quando precisa de resolução; ele roda
  antes do composite de identidade em edições.
- Em JPEGs pequenos de WhatsApp, prefira a saída nativa: ampliar o cenário com
  SeedVR2 e a cabeça deterministicamente produz níveis de detalhe diferentes.
- “Qualquer estilo” não significa um único checkpoint onisciente: o roteador
  usa prompt para estilos comuns e SDXL+LoRA quando um estilo treinado é
  necessário.
- O M5 de 24 GB não transforma PyTorch/MPS em um runtime Metal quantizado. Por
  isso Draw Things é mantido como backend rápido, não desinstalado.

Detalhes: [docs/LIMITES.md](docs/LIMITES.md).

## Licença

Código MIT. Cada peso mantém a licença do seu autor; confira
[docs/MODELOS.md](docs/MODELOS.md) antes de uso comercial.

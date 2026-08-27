# Tutorial de uso

## Pelo CRIAs AI, Claude Code, Codex, OpenCode ou Pi

Peça em linguagem natural e informe o caminho da foto quando houver entrada:

```text
Gere uma foto casual de iPhone de uma cafeteria de bairro à noite.

Edite /Users/eu/Downloads/foto.jpg: troque a camiseta por um moletom preto,
preservando rosto, cabelo, mãos, pose, luz e fundo.

Crie uma cena nova usando /Users/eu/Downloads/rosto.jpg e
/Users/eu/Downloads/roupa.jpg: a pessoa caminhando naturalmente em um museu.
```

O agente escolhe `foto_gerar`, `foto_editar` ou `foto_cena`. Se precisar do
ComfyUI, o conector inicia o serviço automaticamente. A saída padrão fica ao
lado da foto original ou em `~/Downloads`.

## Diagnóstico

```bash
foto status
```

Deve mostrar `ok` para editar, referências, geração rápida e Krea/Famegrid.

## Gerar do zero

Rápido, normalmente entre 40 e 60 segundos:

```bash
foto gerar "foto casual de iPhone de amigos conversando numa cozinha comum" \
  --estilo iphone --saida ~/Downloads/cozinha.png
```

Qualidade máxima/Famegrid — cerca de 3 minutos em 384×512 ou 11,5 minutos em
640×896 neste M5:

```bash
foto gerar "uma fotografia cotidiana, natural e indistinguível de real" \
  --estilo famegrid --tamanho 640x896 \
  --saida ~/Downloads/famegrid.png
```

Outros estilos:

```bash
foto gerar "um gato astronauta" --estilo cartoon
foto gerar "uma cidade futurista" --estilo pixel-art
foto gerar "personagem numa floresta" --estilo anime
foto gerar "frasco de perfume sobre pedra" --estilo produto
```

Se um nome estiver cadastrado no registro privado de identidades, basta usá-lo
no prompt. O roteador seleciona Krea 2, troca o nome pelo token técnico e
empilha a LoRA de identidade com a Famegrid:

```bash
foto gerar "Pessoa apresentando uma palestra de tecnologia, foto casual de iPhone"
```

Veja [IDENTIDADE.md](IDENTIDADE.md). Se a LoRA ainda não existir, o comando
falha explicitamente em vez de gerar silenciosamente outra pessoa.

## Editar uma fotografia existente

Use quando pose e enquadramento devem permanecer iguais:

```bash
foto editar ~/Downloads/foto.jpg \
  "Replace the blue shirt with a plain black hoodie. Keep face, hair, hands, pose, lighting and background unchanged." \
  --saida ~/Downloads/foto_editada.png
```

Fluxo aplicado: Mage-Flow → polimento SDXL → upscale opcional → cabeça original
via Vision/blend multi-banda → casamento de grão. A cabeça original é recolada
por último, depois de qualquer etapa generativa.

Para trocar somente a roupa por descrição:

```bash
foto vestir ~/Downloads/foto.jpg \
  "a white long-sleeved cotton dress shirt without logos"
```

Não passe uma foto de outra pessoa ao `vestir`: o editor pode copiar cabeça,
sapatos ou acessórios. Para pessoa + roupa como referências, use `foto cena`.

## Criar uma cena nova com referências

Primeiro prepare selfies ou retratos:

```bash
foto refs ~/Downloads/selfie1.jpg ~/Downloads/selfie2.heic
```

Depois componha a cena:

```bash
foto cena "candid realistic photo of this person standing in a museum, wearing the referenced outfit" \
  --ref ~/Pictures/refs/id/selfie1_face.png \
  --ref ~/Downloads/corpo.jpg \
  --ref ~/Downloads/roupa.jpg \
  --saida ~/Downloads/museu.png
```

Essa rota usa FLUX.2 multi-reference. Ela preserva características zero-shot,
mas recria a pessoa; não garante identidade biométrica ou pixel-idêntica.

## Ampliar

```bash
foto ampliar ~/Downloads/imagem.png --escala 2
```

SeedVR2 é generativo e pode alterar microdetalhes. Em JPEGs pequenos do WhatsApp
ou quando o rosto já está correto, compare também a versão sem upscale.

## Workflows visuais

Abra `http://127.0.0.1:8188`, recarregue a página e entre em
**Workflows → Browse → foto-macos**:

1. editar foto — Mage-Flow;
2. referências e cena — FLUX.2;
3. gerar versátil — FLUX.2;
4. gerar rápido — Z-Image/Draw Things;
5. fotorrealismo — Krea 2/Famegrid.

Os workflows 4 e 5 chamam os mesmos runtimes Apple do CLI/MCP; não há uma cópia
separada dos modelos apenas para o localhost.

## Escolha rápida

| objetivo | comando/rota |
|---|---|
| alterar algo mantendo a foto | `foto editar` |
| trocar roupa por descrição | `foto vestir` |
| criar imagem sem referência | `foto gerar` |
| máximo fotorrealismo sem pressa | `foto gerar --estilo famegrid` |
| pessoa/roupa/objeto como referências | `foto cena` |
| aumentar resolução | `foto ampliar` |

Sempre inspecione mãos, acessórios, texto, reflexos e pequenos objetos antes de
publicar uma imagem gerada ou uma região editada.

# foto-macos

Edição de foto por instrução, **100 % local**, num MacBook Pro **M5 de 24 GB**.
Você manda a foto e uma frase; o rosto continua sendo o seu.

```bash
foto editar minha-foto.jpg "troque a camiseta por um moletom preto"
```

Não é mais um wrapper de ComfyUI. É um pipeline de cinco estágios em que **cada
etapa existe porque a anterior falhava de um jeito específico e medido** — e a
ordem entre elas foi aprendida errando. Tudo aqui foi cronometrado neste
hardware, não copiado de post.

---

## O problema

Modelos de edição por difusão passam a imagem **inteira** pelo VAE. Mesmo o que
"ficou visualmente igual" tem os pixels reescritos. Medido aqui, numa troca de
camisa por moletom:

| | pixels idênticos à original |
|---|---|
| Qwen-Image-Edit-2511 | **7,3 %** |
| Mage-Flow-Edit-Turbo | **15,8 %** |

Ou seja: seu rosto é *redesenhado* toda vez. Ele sai parecido, e não é você.
E a área redesenhada sai **lisa demais**: numa foto de WhatsApp, o ruído de alta
frequência caiu de **19,45** para **2,04** — quase 10× — e é esse contraste
dentro da mesma foto que o olho lê como "isso é IA".

Este repositório é a resposta a esses dois problemas.

---

## O pipeline

```
 foto original ──┐
                 │
    1. EDITAR    │  Mage-Flow-Edit-Turbo 4B · 4 steps · CFG 1        ~45-160 s
                 │
    2. POLIR     │  SDXL denoise 0.03 + 1x-ITF-SkinDiffDetail          ~130 s
                 │
    3. CABEÇA  ◄─┤  recola a cabeça REAL da foto original          instantâneo
                 │
    4. GRÃO    ◄─┘  injeta o grão que falta na área editada       instantâneo
                 │
    5. AMPLIAR      SeedVR2 2× (MLX)                                    ~26 s
```

**A ordem não é negociável**, e cada regra custou uma imagem ruim:

- **Polir vem antes de recolar a cabeça.** Polir depois regenera o rosto que
  acabou de ser preservado.
- **Ampliar vem por último**, pelo mesmo motivo: o SeedVR2 é generativo e
  reconstrói tudo que enxerga, inclusive o rosto original.
- **Nunca edite uma imagem já editada.** Duas passagens de difusão empilhadas
  achatam o cabelo e deformam o rosto. Edite sempre a foto original.

---

## Instalação

Requer macOS com Apple Silicon, [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
e [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/matheusbgodoi/foto-macos.git
cd foto-macos && ./install.sh
```

O `install.sh` instala as dependências, baixa os modelos e **aplica os patches**
descritos em [docs/BUGS.md](docs/BUGS.md) — sem eles, duas etapas não funcionam
no macOS.

Lista completa de modelos, tamanhos e links: **[docs/MODELOS.md](docs/MODELOS.md)**
(~25 GB no total).

---

## Uso

```bash
# pipeline completo (recomendado)
foto editar foto.jpg "troque o moletom por uma camisa social branca" \
     --saida saida.png

# preparar referências de identidade (auto-orienta e recorta o rosto)
foto refs ~/Downloads/fotos/*.HEIC

# compor cena nova a partir de referências
foto cena "num café, luz natural, foto casual" --ref rosto.png --ref roupa.png

# só ampliar
foto ampliar imagem.png --escala 2
```

Opções úteis: `--sem-cabeca` (quando a pose muda), `--mp 0.5` (rascunho rápido,
~30 s), `--ampliar` (SeedVR2 no fim).

---

## O que funciona, e o que não

Sem marketing:

| caso de uso | estado | nota |
|---|---|---|
| **Editar a sua foto** — trocar roupa, objeto, fundo | funciona | **9/10** |
| **Cena nova a partir de referência**, mantendo o rosto | a cena sai ótima, o rosto sai *parecido* | **4/10** |
| **Gerar do zero** (text-to-image puro) | não montado | — |

**Por que o caso 2 não fecha:** não existe adapter de identidade zero-shot para
nenhum dos modelos abertos atuais. PuLID, InstantID, InfiniteYou e companhia
foram feitos para FLUX.1/SDXL. O único que alega suportar FLUX.2 klein
(`PuLID-Flux2`) carrega os pesos com `strict=False` e os nomes dos tensores não
batem — **toda a camada de identidade é descartada em silêncio e vira ruído
aleatório**. Se você viu anatomia quebrada usando isso, essa é a razão.

Detalhes e outras rotas avaliadas: [docs/LIMITES.md](docs/LIMITES.md).

---

## Medições

Feitas neste M5 de 24 GB, com o modelo já carregado:

| operação | tempo |
|---|---|
| editar, 1 MP | **44–156 s** |
| editar, 2 MP | 226 s |
| polir (SDXL denoise 0.03) | 132 s |
| ampliar SeedVR2 2× | **25,8 s** · pico 11,3 GB |
| ampliar 4x-UltraSharpV2 2× | 183 s |
| recolar cabeça / casar grão | instantâneo |
| **pipeline completo** | **~5 min** |

Resultados de qualidade, na região editada (ruído de alta frequência — quanto
mais perto da original, mais parece foto):

| | original | editado | após polir |
|---|---|---|---|
| camisa | 5,71 | 4,73 | **5,61** |

### O que foi testado e reprovado

| tentativa | resultado |
|---|---|
| Mage-Flow **não-turbo**, 10 steps, CFG 2 | **1767 s** — 13× mais caro, sem ganho visível |
| Mage-Flow não-turbo, 30 steps, CFG 4 | abortado com mais de 1h30 |
| Qwen-Image-Edit-2511 Q4 + Lightning | 341–712 s, e preservou **menos** que o Mage 4B |
| Step1X-Edit | 41,8 GB, sem GGUF, MPS *not planned*, 32 min–4,4 h/imagem |
| Composite 1:1 por diferença de pixels | métrica subia a 90 %, resultado visualmente pior |
| Trocar roupa usando **foto de outra pessoa** como referência | cola partes dela na cena (cabeça flutuando) — descreva a roupa **em texto** |

**Steps altos não compram qualidade aqui.** Turbo de 4 steps com CFG 1 é a
configuração certa neste hardware.

---

## Sobre a comparação com o ChatGPT

**Onde este pipeline ganha:** controle do que *não* muda. Você preserva rosto,
cabelo e fundo como pixels reais — nenhuma ferramenta web oferece isso.
Realismo de pele empata ou ganha (o GPT-Image é reconhecidamente plastificado).

**Onde perde, sem discussão:** aderência a prompt complexo, texto na imagem, e
velocidade — uma GPU dedicada faz em ~1 s o que aqui leva ~2 min.

---

## Créditos

O estágio de polimento veio do
[Qwen Image To Dataset Workflow](https://www.reddit.com/r/comfyui/) (r/comfyui),
de onde saíram os parâmetros de denoise 0.03 e o `1x-ITF-SkinDiffDetail`.
O estágio de upscale seguiu a pista do
[processo de upscale para fotorrealismo](https://civitai.com/images/113122866)
do RaymondLuxuryYacht, que apontava o SeedVR2.

Modelos: [Mage-Flow](https://huggingface.co/Comfy-Org/Mage-Flow) (Microsoft, MIT) ·
[RealVisXL V5](https://huggingface.co/SG161222/RealVisXL_V5.0) ·
[SeedVR2](https://huggingface.co/numz/SeedVR2_comfyUI) via
[mflux](https://github.com/filipstrand/mflux) ·
[RES4LYF](https://github.com/ClownsharkBatwing/RES4LYF) ·
[1x-ITF-SkinDiffDetail](https://openmodeldb.info/models/1x-ITF-SkinDiffDetail-Lite-v1).

MIT.

# Pipeline de foto local — v2 (MacBook Pro M5, 24 GB)

Substitui o pipeline v1 (`../`), que fazia **inpaint com máscara usando um modelo
text-to-image** — abordagem que nunca poderia entregar "muda só o que pedi".

## Como usar

```bash
foto refs    ~/Downloads/fotos/*.HEIC        # prepara referências de identidade
foto editar  FOTO "troque a camiseta por um moletom cinza"
foto vestir  FOTO "an oversized plaid flannel over a white t-shirt"
foto cena    "descrição da cena" --ref rosto.png --ref roupa.png
foto ampliar IMAGEM --escala 2
foto rosto   FOTO_REAL IMAGEM_GERADA SAIDA   # recoloca o rosto real
```

## O que a stack usa

| peça | o quê | por quê |
|---|---|---|
| **Mage-Flow-Edit-Turbo** | DiT 4B, MIT, Microsoft | edição por instrução **sem máscara**; nativo no ComfyUI 0.33 (`TextEncodeMageFlowEdit`, até 16 referências) |
| Qwen3-VL-4B | text/vision encoder | lê as referências e casa a instrução com a foto |
| **SeedVR2** (mflux/MLX) | upscaler generativo | **padrão**; reconstrói microtextura. Não roda no ComfyUI/MPS |
| 4x-UltraSharpV2 | upscaler ESRGAN | alternativa (`--esrgan`); só nitidez de borda |
| Vision.framework | detecção de rosto | nativo do macOS, Neural Engine |
| ComfyUI 0.33 + MPS | runtime | headless em `127.0.0.1:8188` |

## Medições reais nesta máquina

| configuração | tempo | observação |
|---|---|---|
| Mage-Flow-Edit-Turbo, 4 steps, 1 MP | **44–132 s** | **configuração de trabalho** |
| Mage-Flow-Edit-Turbo, 4 steps, 2 MP | 226 s | mais microdetalhe |
| Mage-Flow-Edit **não-turbo**, 10 steps, CFG 2 | **1767 s** | 13× mais caro, **sem ganho visível** |
| Mage-Flow-Edit não-turbo, 30 steps, CFG 4 | >1h30 | abortado; inviável |
| Qwen-Image-Edit-2511 Q4 + Lightning, 4 steps | 341–712 s | pior e mais lento que o Mage |
| **SeedVR2 2x** (896×1216 → 1792×2432) | **25,8 s** | pico 11,3 GB; reconstrói fio de cabelo, poro, barba |
| 4x-UltraSharpV2 → 2x (1184×1776 → 2368×3552) | 183 s | só nitidez de borda |

Fidelidade de edição (trocar camisa por moletom), pixels preservados fora da edição:
Mage cru **15,8 %** · Qwen cru **7,3 %**. Nenhum modelo de difusão preserva pixels:
o VAE re-renderiza o quadro inteiro.

## O que foi tentado e REPROVADO

- **Composite 1:1 por diferença de pixels** (`compose1to1.py`). A métrica subia
  para 90 %, mas o resultado era visualmente pior: colagem parcial, manchas.
  Máscara por diff não separa a edição quando ela é grande e gradual. **Não use.**
- **`vestir` com foto de outra pessoa como referência de roupa.** O modelo cola
  partes da pessoa da referência na cena (cabeça flutuando, sapatos soltos).
  Descreva a roupa **em texto**.
- **Steps altos / CFG real.** Ver tabela: custo 13× sem ganho.
- **MediaPipe** para landmarks: aborta no macOS (`DrishtiMetalHelper: Service is
  unavailable`) e a 1.x removeu `mp.solutions`. Trocado por Vision nativo.
- **Adapters de identidade zero-shot** (PuLID/InstantID/InfiniteYou): não existem
  para nenhum dos modelos atuais. O `PuLID-Flux2` que alega suportar klein está
  quebrado — carrega os pesos com `strict=False` e os nomes dos tensores não
  batem, então a camada de identidade vira ruído aleatório.

## Limites honestos

- **Identidade preservada de verdade** só no modo `editar`/`vestir`, em que a sua
  foto é a base. O modo `cena` **recria** a pessoa: sai parecida, não idêntica.
- Aderência a prompt complexo e texto na imagem perdem para o GPT-Image.
- Velocidade perde de 1 a 2 ordens de grandeza para GPU dedicada.
- Onde **ganha**: controle de preservação do que não foi editado, e realismo de
  pele (o GPT-Image é plastificado).

## Regras que não devem ser mexidas sem medir

1. Referência de identidade = **crop de rosto grande e bem orientado**. Rosto
   pequeno ou foto girada → o modelo descarta a referência silenciosamente.
   `foto refs` resolve os dois (auto-orienta testando as 4 rotações).
2. O Autogrow do `TextEncodeMageFlowEdit` chega pela API como
   **`images.image_1`**, com o prefixo. Sem o prefixo o `dynamic_paths` sai vazio
   e o nó roda **sem nenhuma referência** — falha silenciosa que custa uma imagem
   inteira gerada do zero.
3. O input `vae` do mesmo nó é opcional na assinatura mas **obrigatório na
   prática**: é ele que gera os `ref_latents`.
4. O SeedVR2 do mflux chama `mx.repeat` com um array de contagens em 4 lugares;
   o MLX 0.32.2 só aceita `int`. Há um patch local com o helper `_repeat_var` em
   `mflux/models/seedvr2/model/seedvr2_transformer/attention.py` (backup `.bak`
   ao lado). **Um upgrade do mflux reverte o patch** e o upscale volta a quebrar
   com `TypeError: repeat(): incompatible function arguments`.

## Arquivos

- `foto.py` — CLI (ponto de entrada)
- `mage.py` — runner Mage-Flow-Edit
- `qie.py` — runner Qwen-Image-Edit (mantido para comparação)
- `prepref.py` — prepara referências (auto-orienta + crop de rosto)
- `facedet.py` — detecção de rosto via Vision.framework
- `face1to1.py` — recoloca o rosto real na imagem gerada
- `ampliar.py` — upscale (SeedVR2 padrão, ESRGAN alternativo)
- `upscale.py` — upscale ESRGAN via ComfyUI
- `fidelity.py` — mede preservação de pixels
- `compose1to1.py` — **reprovado**, mantido só como registro

# Modelos e runtimes

O projeto usa modelos especializados. Os links abaixo são as origens dos pesos;
confira a licença de cada versão antes de redistribuir ou usar comercialmente.

## ComfyUI

Todos os caminhos são relativos a `~/comfyui/models/`.

| papel | arquivo | tamanho aproximado | origem |
|---|---|---:|---|
| editar foto | `diffusion_models/mage_flow_edit_turbo_bf16.safetensors` | 7,7 GB | [Comfy-Org/Mage-Flow](https://huggingface.co/Comfy-Org/Mage-Flow) |
| encoder Mage | `text_encoders/qwen3vl_4b_bf16.safetensors` | 8,3 GB | [Comfy-Org/Mage-Flow](https://huggingface.co/Comfy-Org/Mage-Flow) |
| VAE Mage | `vae/mage_flow_vae_bf16.safetensors` | 0,33 GB | [Comfy-Org/Mage-Flow](https://huggingface.co/Comfy-Org/Mage-Flow) |
| gerar/cena multi-ref | `diffusion_models/flux-2-klein-4b.safetensors` | ~8 GB | [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) |
| encoder FLUX.2 | `text_encoders/qwen_3_4b.safetensors` | ~8 GB | [Comfy-Org/z_image_turbo](https://huggingface.co/Comfy-Org/z_image_turbo) |
| VAE FLUX.2 | `vae/flux2-vae.safetensors` | ~0,3 GB | [Comfy-Org/flux2-dev](https://huggingface.co/Comfy-Org/flux2-dev) |
| gerar SDXL/polir | `checkpoints/RealVisXL_V5.0_fp16.safetensors` | 6,5 GB | [SG161222/RealVisXL_V5.0](https://huggingface.co/SG161222/RealVisXL_V5.0) |
| textura | `upscale_models/1x-ITF-SkinDiffDetail-Lite-v1.pth` | 19 MB | [OpenModelDB](https://openmodeldb.info/models/1x-ITF-SkinDiffDetail-Lite-v1) |

Mage-Flow é o editor da foto-base; FLUX.2 Klein é o modelo unificado de geração
e multi-referência. RealVisXL serve ao ecossistema SDXL/LoRA e ao polimento de
denoise 0,03.

Os grafos Mage/FLUX usam nós nativos da versão atual do ComfyUI. Os dois
workflows que chamam runtimes Apple externos (Draw Things e Krea/MLX) usam o
custom node fino em `integrations/comfyui`; ele apenas descarrega pesos MPS,
executa o mesmo runner do CLI/MCP e devolve uma `IMAGE` ao canvas.

## Draw Things

Backend opcional e preferido quando disponível para geração rápida:

- runtime: [Draw Things](https://github.com/drawthingsai/draw-things-community)
- modelo: [Z-Image Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo), variante i8x local
- diretório padrão desta máquina: `~/Library/Application Support/local-photo-ai-m5/models/`

O instalador não duplica automaticamente esses pesos. Sem eles, o roteador usa
FLUX.2 no ComfyUI. Mantemos o Draw Things porque seu runtime Metal quantizado é
mais apropriado ao Apple Silicon que os mesmos pesos BF16 via PyTorch/MPS.

## MLX

[MFLUX](https://github.com/mflux-community/mflux) executa dois componentes:

| papel | modelo | armazenamento | licença/origem |
|---|---|---:|---|
| fotorrealismo | Krea 2 Turbo Q4 | ~15,8 GB | [MFLUX Q4](https://huggingface.co/mflux-community/krea-2-turbo-mflux-q4), herdando Krea 2 Community License |
| estética natural | Famegrid Natural V1 (LoKr) | ~1,5 GB | [Civitai 2088956/3248281](https://civitai.com/models/2088956?modelVersionId=3248281) |
| ampliar | SeedVR2 | ~12 GB | baixado pelo MFLUX no primeiro uso |

O Krea 2 oficial recomenda treinar LoRAs no Raw e inferir no Turbo; o Famegrid
declara base Krea 2 e usa a trigger `Famegrid`. O peso sugerido pelo autor é
0,3–1,0; o padrão local é 0,7. Veja [KREA2.md](KREA2.md).

SeedVR2 é generativo: em edição ele roda antes do composite da cabeça; se
falhar, o fallback é Lanczos.

## Não incluído no fluxo vencedor

| candidato | veredito medido/estrutural |
|---|---|
| Qwen-Image-Edit-2511 Q4 | 341–712 s e mais drift que Mage neste M5 |
| Step1X-Edit | CUDA/Triton, sem rota MPS suportada e memória incompatível |
| Mage não-turbo | 1.767 s em 10 steps/CFG 2 sem ganho visível |
| 4x-UltraSharpV2 | removido do instalador de produção por licença não comercial |
| PuLID-Flux2 não oficial | não usado; integração/pesos não demonstraram carga válida |

Os pesos descartados foram removidos da instalação ativa. A cópia BF16 do
Z-Image no ComfyUI também não é necessária: o workflow visual chama o mesmo
Draw Things i8x usado pelo CLI/MCP.

# Modelos

Total ~25 GB. Todos os caminhos são relativos a `~/comfyui/models/`.

## Estágio 1 — editar

| arquivo | tamanho | destino | origem |
|---|---|---|---|
| `mage_flow_edit_turbo_bf16.safetensors` | 7,7 GB | `diffusion_models/` | [Comfy-Org/Mage-Flow](https://huggingface.co/Comfy-Org/Mage-Flow) |
| `qwen3vl_4b_bf16.safetensors` | 8,3 GB | `text_encoders/` | idem |
| `mage_flow_vae_bf16.safetensors` | 0,33 GB | `vae/` | idem |

**Mage-Flow-Edit-Turbo** (Microsoft, 4B, MIT) é o editor por instrução. Suporte
**nativo** no ComfyUI ≥ 0.33 — nenhum custom node necessário. O nó
`TextEncodeMageFlowEdit` aceita até 16 imagens de referência e devolve
`positive`, `negative` e o `latent` já no tamanho certo.

Por que ele e não os 20B: as referências são redimensionadas para a resolução de
saída antes de entrar no RoPE, o que segura o drift. E, medido aqui, ele é 4×
mais rápido que o Qwen-Image-Edit-2511 **e** preserva mais da foto original.

Variantes opcionais do mesmo repo: `mage_flow_edit_bf16` (30 steps — testado,
**não compensa**: 13× mais caro sem ganho) e `mage_flow_turbo_bf16` (text-to-image,
ainda não integrado).

## Estágio 2 — polir

| arquivo | tamanho | destino | origem |
|---|---|---|---|
| `RealVisXL_V5.0_fp16.safetensors` | 6,5 GB | `checkpoints/` | [SG161222/RealVisXL_V5.0](https://huggingface.co/SG161222/RealVisXL_V5.0) |
| `1x-ITF-SkinDiffDetail-Lite-v1.pth` | 19 MB | `upscale_models/` | [openmodeldb](https://openmodeldb.info/models/1x-ITF-SkinDiffDetail-Lite-v1) · [uwg/upscaler](https://huggingface.co/uwg/upscaler) |

Qualquer checkpoint SDXL fotorrealista serve. O workflow original usava
`epicrealismXL`; aqui foi escolhido o RealVisXL V5 por ser SFW e de qualidade
equivalente para esse fim — o estágio só reescreve altíssima frequência, então
o que importa é o realismo de pele e tecido do checkpoint.

## Estágio 5 — ampliar

**SeedVR2** roda via [mflux](https://github.com/filipstrand/mflux) (MLX), **não**
pelo ComfyUI — no ComfyUI/MPS ele consome 73–88 GB (issues 15053 e 15785, ambas
abertas). O mflux baixa os pesos sozinho de
[numz/SeedVR2_comfyUI](https://huggingface.co/numz/SeedVR2_comfyUI) (~12 GB) no
primeiro uso.

```bash
uv tool install mflux
```

Alternativa não-generativa (`--esrgan`), útil quando você **não** quer que nada
seja reinventado:

| arquivo | tamanho | destino | origem |
|---|---|---|---|
| `4x-UltraSharpV2.safetensors` | 133 MB | `upscale_models/` | [Kim2091/UltraSharpV2](https://huggingface.co/Kim2091/UltraSharpV2) |

## Detecção de rosto

Nenhum download. Usa o **Vision.framework** do próprio macOS, via
`pyobjc-framework-Vision` — roda no Neural Engine. Ver
[BUGS.md](BUGS.md) para o motivo de não ser MediaPipe.

## Custom nodes

| node | necessário? | para quê |
|---|---|---|
| [RES4LYF](https://github.com/ClownsharkBatwing/RES4LYF) | opcional | samplers `res_2s` e afins |
| [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) | opcional | só se quiser testar modelos quantizados |
| [ComfyUI-Manager](https://github.com/Comfy-Org/ComfyUI-Manager) | opcional | conveniência |

O pipeline principal **não depende de nenhum custom node**.

## Avaliados e descartados

| modelo | por quê |
|---|---|
| Qwen-Image-Edit-2511 (20B) | 341–712 s/imagem e preservou menos que o Mage 4B |
| Step1X-Edit (~20B) | 41,8 GB, sem GGUF, MPS marcado *not planned*, 32 min–4,4 h/imagem |
| Krea 2 | fp8 é bloqueado em MPS por código; bf16 são 26 GB só de UNet |
| PuLID-Flux2 | quebrado por construção — ver [LIMITES.md](LIMITES.md) |

# Bugs encontrados no caminho

Quatro coisas quebradas que custaram horas. Se você está montando algo parecido
no macOS, provavelmente vai bater nas mesmas.

---

## 1. O nó do Mage-Flow ignora as referências pela API — em silêncio

**Sintoma:** você manda a foto como referência, o modelo gera uma imagem
totalmente do zero e ninguém reclama. Nenhum erro, nenhum aviso. A saída sai em
1024×1024 (o default) em vez do aspecto da sua foto — o único indício.

**Causa:** o input `images` de `TextEncodeMageFlowEdit` é um *Autogrow*. Pela API
ele chega **achatado e com o id do slot como prefixo**:

```jsonc
// errado — dynamic_paths sai vazio e o nó roda sem referência nenhuma
{ "image_1": ["201", 0] }
{ "images": { "image_1": ["201", 0] } }

// certo
{ "images.image_1": ["201", 0] }
```

O `execution.py` re-aninha via `build_nested_inputs()` usando `dynamic_paths`.
Sem o prefixo `images.`, o mapa sai vazio e o nó executa com zero referências.

Para descobrir o nome esperado de qualquer Autogrow:

```python
from comfy_api.latest import _io
vi, hidden, v3 = _io.get_finalized_class_inputs(cls.INPUT_TYPES(), live_inputs)
print(v3.get("dynamic_paths"))
```

**Bônus:** o input `vae` é `optional=True` na assinatura mas é **obrigatório na
prática** — é ele que gera os `ref_latents`.

---

## 2. MediaPipe aborta o processo no macOS

**Sintoma:**

```
F0000 graph_service.h:139] Check failed: service_ Service is unavailable.
*** Check failure stack trace: ***
    @  -[DrishtiMetalHelper initWithCalculatorContext:]
```

O processo morre inteiro, não é exceção capturável. Acontece dentro do *face
detector*, no caminho Metal. Forçar `delegate=CPU` nas `BaseOptions` **não
resolve** — o detector usa Metal de qualquer forma. E o MediaPipe 1.x removeu
`mp.solutions`, então o caminho legado em CPU não existe mais.

**Solução:** `Vision.framework`, que já vem no macOS.

```bash
uv pip install pyobjc-framework-Vision pyobjc-framework-Quartz
```

Roda no Neural Engine, sem download de modelo. Uma pegadinha: os pontos vêm num
`objc.varlist` (ponteiro C sem tamanho) e precisam ser fatiados com o
`pointCount`; e a origem é no canto **inferior** esquerdo.

```python
raw = region.normalizedPoints()[:region.pointCount()]
```

Ver `src/facedet.py`.

---

## 3. SeedVR2 no mflux quebra com o MLX 0.32

**Sintoma:**

```
TypeError: repeat(): incompatible function arguments.
  1. repeat(array, repeats: int, axis: int | None = None, ...)
Invoked with types: mlx.core.array, mlx.core.array, kwargs = { axis: int }
```

**Causa:** o código chama `mx.repeat(x, mx.array(counts), axis=0)` — contagens
variáveis por elemento — em **4 lugares** de
`mflux/models/seedvr2/model/seedvr2_transformer/attention.py`. O MLX 0.32.2 só
aceita `repeats: int`.

**Solução:** um helper que repete fatia a fatia e concatena (mesma semântica de
`np.repeat` com lista). O `install.sh` aplica o patch e guarda um `.bak`.

⚠️ **Um upgrade do mflux reverte o patch** e o upscale volta a quebrar.

---

## 4. Nada disso é CUDA — e quase tudo que a comunidade recomenda é

Descartados por dependência CUDA ou por estarem quebrados em MPS:

| ferramenta | problema |
|---|---|
| Nunchaku / SVDQuant | CUDA-only |
| SeedVR2 **no ComfyUI** | 73–88 GB de consumo (issues 15053, 15785, abertas) |
| Step1X-Edit | `.to("cuda")` hardcoded; `liger_kernel` → `triton`, sem wheel para macOS ARM |
| fp8 (`float8_e4m3fn`) | `model_management.py::supports_cast()` retorna `False` para MPS antes de olhar o dtype |
| insightface + onnxruntime-gpu | base de quase todo adapter de identidade |

**Flags do ComfyUI que importam no Mac:**

```bash
python main.py --use-pytorch-cross-attention --reserve-vram 2 --listen 127.0.0.1
```

Evite `--force-fp16`: ele é alias de `--fp16-unet` e força `float16` **antes** de
consultar o modelo. O Mage-Flow declara só `[bfloat16, float32]`. E
`--reserve-vram 6` é excessivo: em MPS o `get_free_memory()` lê
`psutil.available`, e reservar 6 GB de ~16 GB úteis provoca ciclos de
offload/reload.

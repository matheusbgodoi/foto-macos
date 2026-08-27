# Identidade local no Krea 2

O fluxo treina uma LoRA no **Krea 2 Raw** e a aplica no **Krea 2 Turbo** junto
da Famegrid. Fotos, captions, UUIDs do Apple Photos, checkpoints e a LoRA final
ficam fora do repositório.

Uma LoRA melhora muito a repetibilidade de rosto, cabelo e aparência sem exigir
uma imagem de referência em todo pedido. Ela não é uma garantia biométrica:
mãos, acessórios, idade, perfil extremo e cenas com várias pessoas ainda devem
ser inspecionados.

## 1. Preparar um dataset do Apple Photos

O utilitário consulta a biblioteca em modo somente leitura, exige um único rosto
detectado e balanceia retratos próximos, médios e ambientais. Use um token raro;
o nome amigável será configurado depois.

```bash
PRIVATE="$HOME/Library/Application Support/foto-macos/identity/pessoa"

python src/prepare_photos_identity.py \
  --library "$HOME/Pictures/Photos Library.photoslibrary" \
  --person "nome no álbum Pessoas" \
  --trigger pessoa_rara_token \
  --output "$PRIVATE" \
  --count 40
```

Revise `contact-sheet.jpg` e **todas as captions** antes de treinar. Remova fotos
com filtros, texto sobreposto, outras pessoas, reflexos duplicados, rosto oculto
e várias imagens quase iguais. As captions automáticas são rascunhos porque o
Photos não descreve roupa ou cenário: acrescente roupa, óculos, pose, expressão,
iluminação e ambiente como variáveis. Não descreva rosto, cabelo ou outros
traços permanentes que o token deve aprender. O script aceita
`--reject-uuid-file` para repetir a seleção sem alterar o app Fotos.

## 2. Obter o checkpoint de treino

Aceite os termos de [Krea-2-Raw](https://huggingface.co/krea/Krea-2-Raw) na
mesma conta usada por `hf auth login`. O Turbo não substitui o Raw durante o
treino; a própria Krea recomenda treinar no Raw e aplicar a LoRA no Turbo.

O MFLUX é uma opção de treino no próprio Apple Silicon:

```bash
uv tool install mflux
mflux-train --config "$PRIVATE/train.json" --dry-run
mflux-train --config "$PRIVATE/train.json"
```

Comece com uma época piloto. Verifique que os tensores `lora_B` do checkpoint
possuem norma diferente de zero e gere exemplos com seeds fixas antes de rodar
o treino completo. Em 24 GB use QLoRA 8-bit, `max_resolution: 512`, batch 1,
gradient checkpointing nativo do adaptador Krea e `low_ram: true`. Para datasets
maiores ou 768 px, uma GPU NVIDIA de 24 GB é mais prática.

### RTX 3090 + SimpleTuner

Krea 2 Raw tem 26,98 GB lógicos, portanto não cabe inteiro em BF16 numa placa de
24 GB. A configuração validada usa `int8-torchao` com quantização inicial na
CPU e executa forward/backward em CUDA BF16. Em 768 px, batch 1 e rank 32, os
gradientes/estado do otimizador são descarregados para RAM; o modelo e o cálculo
principal continuam na GPU.

Opções relevantes:

```json
{
  "base_model_precision": "int8-torchao",
  "quantize_via": "cpu",
  "caption_dropout_probability": 0.0,
  "mixed_precision": "bf16",
  "resolution": 768,
  "train_batch_size": 1,
  "lora_rank": 32,
  "gradient_checkpointing": true,
  "fuse_qkv_projections": true,
  "attention_mechanism": "cudnn",
  "optimizer": "optimi-lion",
  "optimizer_offload_gradients": true,
  "optimizer_release_gradients": false,
  "optimizer_config": "kahan_sum=false,foreach=false",
  "max_grad_norm": 0.01
}
```

Mantenha `caption_dropout_probability` em zero para identidade: o exemplo
oficial Krea 2 do SimpleTuner também usa zero, evitando treinar imagens da pessoa
sem o token que deve acioná-la.

Não ative `optimizer_release_gradients` nessa combinação: o treino pode concluir
e salvar um adapter cujos tensores `lora_B` continuam zerados. Valide sempre as
normas do primeiro checkpoint antes de iniciar uma execução longa. Desativar
Kahan e `foreach` remove o estado temporário que estourava os 24 GB sem descartar
os gradientes.

### SimpleTuner com QKV fundido + MFLUX

O exemplo oficial do SimpleTuner usa `fuse_qkv_projections=true`; por isso o
adapter PEFT contém `attn.to_qkv`. O Krea 2 no MFLUX representa a mesma atenção
como `to_q`, `to_k` e `to_v` separados. Converta o checkpoint antes da
inferência no Mac:

```bash
python src/convert_krea2_fused_qkv_lora.py \
  checkpoint/pytorch_lora_weights.safetensors \
  identidade-krea2-mflux.safetensors
```

A conversão não aproxima nem reprocessa os pesos: ela reutiliza a projeção A e
divide a projeção B nas dimensões exatas de Q/K/V (incluindo o GQA 48/12/12 do
transformer). Confirme no log do MFLUX que não restaram chaves `to_qkv` sem
correspondência.

## 3. Cadastrar o nome amigável

Crie o arquivo privado:

`~/Library/Application Support/foto-macos/identities.json`

```json
{
  "Nome": {
    "token": "pessoa_rara_token",
    "lora": "/caminho/privado/identidade-krea2.safetensors",
    "scale": 0.85,
    "famegrid_scale": 0.3
  }
}
```

Depois disso, CLI e MCP reconhecem o nome sem opção extra:

```bash
foto gerar "Nome trabalhando naturalmente em um café, foto casual de iPhone"
```

O roteador executa Krea 2 Turbo + Famegrid + LoRA de identidade. Sem identidade,
Famegrid usa 0,7; com identidade, cai para 0,3 por padrão para não dominar os
traços aprendidos. `--peso` sobrescreve esse valor. `foto status` mostra `ok` ou
`PENDENTE` para cada identidade cadastrada.

## Privacidade

- Não coloque o diretório privado dentro do clone Git.
- Não publique dataset, UUIDs, captions, checkpoints ou LoRA sem consentimento.
- O utilitário não apaga, edita nem favorita itens no Apple Photos.
- Prefira fotos adultas e exclua qualquer imagem com outra pessoa, mesmo ao
  fundo, para não misturar identidades.

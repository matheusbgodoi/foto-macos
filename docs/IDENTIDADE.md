# Identidade local no Krea 2

O fluxo treina uma LoRA no **Krea 2 Raw** e a aplica no **Krea 2 Turbo**. A
Famegrid é opcional e deve ser ligada somente depois de um teste A/B, porque uma
LoRA de estilo pode competir com a identidade. Fotos, captions, UUIDs do Apple
Photos, checkpoints e a LoRA final ficam fora do repositório.

Uma LoRA melhora muito a repetibilidade de rosto, cabelo e aparência sem exigir
uma imagem de referência em todo pedido. Ela não é uma garantia biométrica:
mãos, acessórios, idade, perfil extremo e cenas com várias pessoas ainda devem
ser inspecionados.

## Escolha do modelo-base

| base | identidade | realismo | custo local | decisão |
|---|---|---|---|---|
| **Krea 2 Raw → Turbo** | forte quando o checkpoint é escolhido antes do overfit | melhor estética cotidiana do conjunto | treino pesado, inferência Turbo em 8 passos | principal para identidade + fotografia natural |
| **FLUX.1-dev** | madura e comprovada; um treino anterior nesta 3090 chegou a 0,754 de similaridade ArcFace | bom, mas tende ao visual mais produzido | treino e inferência mais lentos que modelos pequenos | controle/fallback se Krea não superar a identidade já medida |
| **SDXL** | funciona, mas costuma exigir mais ajuste e perde traços finos com facilidade | depende muito do checkpoint | mais leve e enorme ecossistema de LoRAs | estilos e polimento, não identidade principal |

Krea 2 não é “automaticamente melhor” em rosto: ele foi escolhido pelo teto de
realismo, e só permanece como vencedor se superar o controle FLUX na avaliação
objetiva e visual. Em cenas com muitas pessoas, uma LoRA Krea forte pode afetar
os outros rostos; nesses casos prefira geração em duas etapas ou edição local.

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

Treinar Krea 2 no Apple Silicon ainda não é a rota recomendada pelos projetos de
treino. O Mac de 24 GB é o runtime de inferência; use uma NVIDIA de 24 GB para o
treino em resolução final. A RTX 3090 cabe com o modelo-base em FP8 e o adapter
treinável em BF16. Quantizar a base **não** transforma a LoRA final em 8-bit nem
4-bit.

### Dataset que funciona para identidade

Qualidade e cobertura vencem volume bruto. Um ponto de partida reproduzível é:

- 15–30 fotos realmente diferentes; não use 100 variações quase iguais;
- 50–60% de cabeça/ombros, com o rosto ocupando aproximadamente 10–30% do
  quadro;
- 20–30% de planos médios e 2–5 fotos de corpo para aprender proporções;
- frente, 3/4 e perfil, expressões e luzes diferentes;
- sem filtros de beleza, rosto coberto, reflexos duplicados, outra pessoa ou
  compressão destrutiva;
- roupa, óculos, pose, fundo e iluminação descritos em cada caption;
- rosto, cabelo e traços permanentes **não** descritos: o token raro deve
  aprender justamente essas características.

Uma foto de terno não “gruda” o terno quando ele é descrito na caption e o
restante do dataset varia a roupa. Acrescentar imagens só ajuda quando traz
informação nova e limpa.

### RTX 3090 + Musubi Tuner

O [Musubi Tuner](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/krea2.md)
tem uma implementação específica do Krea 2 e permite descarregar blocos do DiT
sem descarregar a GPU inteira. Esta receita foi medida numa RTX 3090: cerca de
15 GB de VRAM, 5 s/passo em 768 px e GPU próxima de 100%.

O repositório inclui `scripts/train-krea2-identity.sh` para criar os caches e
executar a receita abaixo sem gravar fotos ou pesos no clone.

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
  src/musubi_tuner/krea2_train_network.py \
  --dit /modelos/krea-2/raw.safetensors \
  --vae /modelos/qwen_image_vae.safetensors \
  --dataset_config dataset.toml \
  --sdpa --mixed_precision bf16 \
  --timestep_sampling krea2_shift --weighting_scheme none \
  --optimizer_type adamw8bit --learning_rate 1e-4 \
  --gradient_checkpointing \
  --network_module networks.lora_krea2 \
  --network_dim 32 --network_alpha 32 \
  --fp8_base --fp8_scaled \
  --blocks_to_swap 16 --block_swap_h2d_only --block_swap_ring_size 2 \
  --max_train_steps 2000 --save_every_n_steps 250 \
  --output_dir output --output_name identidade-krea2
```

Faça antes os dois caches documentados pelo Musubi:
`krea2_cache_latents.py` com o VAE e
`krea2_cache_text_encoder_outputs.py` com o Qwen3-VL. O uso de
`krea2_shift` é intencional: ele ajusta o timestep ao bucket de cada foto.
Evite `torch.compile` no primeiro treino; ele acrescenta pressão de memória e
tem custo alto de compilação para os vários formatos do dataset.

Salve e gere amostras a cada 250 passos. Compare pelo menos um close frontal e
uma cena distante, sempre com a mesma seed. O melhor checkpoint pode estar em
750, 1000, 1500 ou depois; não promova o último automaticamente. Krea 2 pode
memorizar enquadramento e alisar o rosto quando passa do ponto.

Use ArcFace/InsightFace apenas como sinal auxiliar: compare o candidato com um
centro robusto de várias referências e sempre faça aprovação visual. Uma única
selfie como referência distorce a avaliação por luz, óculos e ângulo.

### Alternativas e limitações

O [exemplo oficial do Diffusers](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_krea2.md)
também recomenda rank/alpha 32 e 1.000 passos. O SimpleTuner documenta
`int8-torchao` em 1024, batch 1, perto do limite de uma GPU de 24 GB. Ambos são
válidos, mas na RTX 3090 testada o AI Toolkit com PyTorch 2.13 consumiu quase
toda a VRAM e ficou mais de oito minutos sem completar o primeiro passo; a rota
Musubi acima completou 24 passos em 120 segundos.

LoRA de identidade e LoRA de estilo devem ser validadas separadamente antes de
empilhar. Há um [relato técnico](https://github.com/ostris/ai-toolkit/issues/964)
de LoRAs comuns do Krea 2 degradando ao serem combinadas e um
[relato de vazamento de identidade](https://github.com/krea-ai/krea-2/issues/15)
em cenas com várias pessoas. `foto gerar --sem-famegrid` isola a identidade.

### Pessoa recorrente vs. pessoa arbitrária por referência

São dois problemas diferentes e não devem compartilhar um comando silenciosamente:

- uma LoRA pessoal associa um token raro a uma identidade recorrente. É o caminho
  mais rápido quando a pessoa é usada com frequência e não exige mandar uma foto
  a cada geração;
- o [Krea 2 ReID Reference](https://huggingface.co/yijunwang2/krea2-reid) é um
  adapter funcional para uma pessoa arbitrária. Ele recebe exatamente uma imagem
  de referência e combina condicionamento visual do Qwen3-VL, tokens limpos do VAE,
  atenção isolada e cache K/V. Não funciona se for carregado como uma LoRA de estilo
  comum.

O caminho ReID exige o workflow e a revisão fixados pelo autor, além dos nodes
[`ComfyUI-Krea2-Ostris-Edit`](https://github.com/ostris/ComfyUI-Krea2-Ostris-Edit).
Os parâmetros testados são Krea 2 Turbo, 8 passos, CFG 1 no ComfyUI, peso 1 e
referência preparada em até `384 * 384` pixels. O checkpoint BF16 com referência
passou de 31 GiB no teste publicado. O INT8 ConvRot também não é uma rota Apple:
no M5 testado, os pesos carregaram, mas `aten::_int_mm` não existe no MPS; o
fallback para CPU chegou a aproximadamente 33 GB de swap e não completou o
primeiro dos oito passos em 176 s. O workflow 06 fica disponível no ComfyUI para
hosts CUDA, mas não entra no roteador de produção do Mac.

### Quando testar LoKr

Se a LoRA padrão variar demais entre seeds mesmo depois de escolher o melhor
checkpoint, LoKr é o próximo experimento controlado. MFLUX consegue carregar o
layout Krea 2 LoKr exportado pelo AI Toolkit, mas a detecção automática de
arquitetura LoKr do Musubi ainda não lista Krea 2; use AI Toolkit/SimpleTuner
nesse experimento em vez de apenas trocar `--network_module` na receita acima.
Relatos comunitários apontam maior retenção de detalhes com `factor=16`, enquanto
outros não encontram ganho consistente. Portanto LoKr só substitui LoRA depois de
um A/B com o mesmo dataset, prompts, seeds e runtime; o nome da técnica não é
evidência de melhor identidade.

No SimpleTuner testado, a configuração que realmente atualizou os pesos numa
RTX 3090 foi: `int8-torchao`, 768 px, `factor=16`, `full_matrix=true`,
`bypass_mode=true`, apenas `Krea2Attention`, QKV **não** fundido e
`optimizer_release_gradients=false`. Incluir `Krea2SwiGLU` quebra o caminho
LyCORIS; liberar os gradientes produziu um arquivo válido com todos os fatores
efetivos zerados; manter QKV fundido sem a liberação falhou no backward do
TorchAO. Esses são bugs/limites do runtime, não ajustes estéticos.

Converta o checkpoint LyCORIS para os nomes esperados pelo MFLUX:

```bash
python src/convert_krea2_lycoris_lokr.py \
  checkpoint/pytorch_lora_weights.safetensors \
  identidade-krea2-mflux.safetensors
```

A conversão aceita Q/K/V separado e também divide projeções fundidas nas
dimensões exatas de Q/K/V. Ela preserva os fatores Kronecker sem reconstruir ou
aproximar o adapter, recusa uma arquitetura desconhecida e falha se todos os
fatores treináveis estiverem zerados. Antes de um treino longo, faça um smoke
test e exija que o MFLUX reporte todas as chaves aplicadas.

### Resultado do treino de referência (M5 + RTX 3090)

O treino reproduzível deste repositório foi avaliado com 24 fotos de 2025–2026,
duas seeds fixas, o mesmo prompt e Krea 2 Turbo Q4. A aprovação visual da pessoa
teve precedência sobre ArcFace:

| candidato | peso | similaridade observada | decisão |
|---|---:|---:|---|
| LoRA Musubi, step 750 | **1,1** | 0,702 / 0,679 nas duas seeds | **produção; aprovado visualmente** |
| LoRA Musubi, step 1750 | 1,1 | 0,753 / 0,738 | métrica alta, preservado para comparação; não promovido sem aprovação visual |
| LoRA Musubi, step 2000 | 1,1 | 0,684 | rejeitado: rosto artificial/plástico |
| LoKr SimpleTuner, step 250/500/750 | 1,1 | 0,105 / 0,104 / 0,109 | rejeitado: a mesma pessoa genérica em todos os checkpoints |

No LoKr, 480/480 chaves foram aplicadas e a escala `alpha` foi conferida contra
o código do LyCORIS; não era falha de conversão. O regime estável nessa RTX 3090
conseguiu atingir apenas atenção, enquanto a LoRA Musubi vencedora também
treinou os lineares feed-forward. A configuração completa do LyCORIS falhou
nesse caminho, portanto “LoKr” não foi tratado como sinônimo de mais qualidade.

Peso 1,4 também foi rejeitado: intensidade maior passou do ponto útil e começou
a deformar geometria, cabelo e expressão. O registro privado usa step 750,
peso 1,1 e `famegrid_scale: 0`; identidade e estilo são empilhados somente
depois de um A/B explícito.

Não acrescente imagens automaticamente quando um checkpoint já foi aprovado.
As 24 fotos atuais cobrem frente, 3/4, perfil, expressões, luzes e planos de
corpo. Uma rodada futura só deve adicionar 4–8 retratos realmente novos e
limpos se várias seeds/prompts demonstrarem uma lacuna específica.

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

O roteador executa Krea 2 Turbo + LoRA de identidade e, opcionalmente, Famegrid.
Sem identidade, Famegrid usa 0,7; com identidade, cai para 0,3 por padrão para
não dominar os traços aprendidos. Use `"famegrid_scale": 0` no registro para
desligá-la inclusive no prompt, `--peso` para sobrescrever o valor ou
`--sem-famegrid` num teste isolado. `foto status` mostra `ok` ou `PENDENTE` para
cada identidade cadastrada.

Identidade e estilo são decisões separadas: mesmo com `--estilo iphone`,
`profissional` ou `foto-natural`, um nome cadastrado força o backend Krea 2. O
preset continua controlando apenas a aparência fotográfica.

## Privacidade

- Não coloque o diretório privado dentro do clone Git.
- Não publique dataset, UUIDs, captions, checkpoints ou LoRA sem consentimento.
- O utilitário não apaga, edita nem favorita itens no Apple Photos.
- Prefira fotos adultas e exclua qualquer imagem com outra pessoa, mesmo ao
  fundo, para não misturar identidades.

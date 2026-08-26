# Krea 2 + Famegrid no Mac

## Por que entrou agora

Krea 2 é um modelo denso de aproximadamente 13B. O checkpoint Turbo BF16 e seus
componentes somam cerca de 33 GB no primeiro download, portanto não era uma boa
rota em PyTorch/MPS num Mac de 24 GB. O suporte MLX do MFLUX e o snapshot Q4
mudam essa conta: o pacote pronto ocupa ~15,8 GB e roda em 8 passos.

O projeto oficial distribui:

- **Raw**: base maleável para treino; inferência oficial usa 52 steps/CFG 3,5;
- **Turbo**: destilado para inferência; 8 steps e CFG desativado;
- recomendação oficial: treinar LoRA no Raw e aplicar no Turbo.

Origem: [krea-ai/krea-2](https://github.com/krea-ai/krea-2). Os pesos usam a
Krea 2 Community License; uma quantização conserva a mesma licença.

## O que há por trás dos samples Famegrid

Os PNGs do pacote de exemplos carregam metadados verificáveis. O primeiro
sample usa 1440×1920 e:

```text
Krea 2 Raw
fase 1: 6 steps · multistep/res_2m · beta57
fase 2: 2 steps · multistep/deis_3m · bong_tangent
FameGrid style: ultrareal3 · strength 1
```

Os prompts também são muito detalhados: composição, pose, materiais, luz,
imperfeições ópticas, textura de pele e objetos do ambiente. Portanto a galeria
não demonstra “só um LoRA”: é base + adaptação natural + prompt expansion +
amostragem em duas fases.

## Modo local

```bash
hf download mflux-community/krea-2-turbo-mflux-q4
foto gerar "uma foto casual numa cozinha comum" --estilo famegrid
```

O modo local usa Krea 2 Turbo Q4, 8 steps, guidance 1,0 e Famegrid 0,7.
O workflow do localhost **Fotorrealismo — Krea 2 Famegrid MLX** chama o mesmo
runner, não duplica os pesos no diretório do ComfyUI.

`guidance=0` não é equivalente a “CFG desligado” neste runtime: ele anulou a
condicionante e fez o modelo ignorar um prompt fotográfico inteiro. O valor
correto do Turbo no MFLUX é 1,0.

Esta rota é somente texto-para-imagem. O snapshot MFLUX Q4 ainda não implementa
edição ou referência para Krea 2; essas operações continuam em Mage-Flow e
FLUX.2, respectivamente.

## Medição no M5 de 24 GB

Mesmo prompt Famegrid e mesma seed:

| motor | resolução | tempo de parede | pico MLX |
|---|---:|---:|---:|
| Krea 2 Q4 + Famegrid | 384×512 | 172,6 s | 18,15 GB |
| Krea 2 Q4 + Famegrid | 640×896 | 689,1 s | 18,15 GB |
| Z-Image i8x / Draw Things | 640×896 | 48,8 s | — |

O Krea foi **14,1× mais lento** na comparação cheia. Ele produziu uma aparência
mais cotidiana e menos cinematográfica, mas não elimina a inspeção de mãos e
objetos. Por isso `auto` só o seleciona quando o pedido diz `Famegrid`, `Krea 2`,
`qualidade máxima`, `teto de realismo` ou `indistinguível de real`; o Z-Image
continua sendo a rota rápida.

O workflow 6+2 do autor deve ser tratado como modo “studio”: a receita depende
de samplers/schedulers específicos e precisa ser medida neste M5 antes de virar
o padrão. Não se presume que uma imagem da galeria seja reproduzível só com os
nomes dos parâmetros.

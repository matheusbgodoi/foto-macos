# Comparação dos modelos e runtimes

Medições feitas no MacBook Pro M5 base com 24 GB de memória unificada. Os
tempos são observados nesta instalação, não números publicados pelos autores.
Resolução, número de referências, pressão de memória e primeiro carregamento
podem alterar bastante o resultado.

## Componentes ativos

| modelo/componente | tipo | para que serve no `foto-macos` | quando é escolhido | tempo observado | principal limite |
|---|---|---|---|---:|---|
| **Z-Image Turbo i8x** | gerador de imagem | geração rápida de fotografia e estilos comuns | padrão de `foto gerar`, via Draw Things | **40–60 s** perto de 1 MP; 54,7 s no teste 768×1024 | menos natural que Krea/Famegrid e ainda pode errar anatomia |
| **Krea 2 Turbo Q4** | gerador de imagem | maior teto de fotorrealismo e estética cotidiana | `--estilo famegrid`, `--motor krea2` ou pedido explícito por qualidade máxima | **172,6 s** em 384×512; **689,1 s** em 640×896 | 3–14× mais lento que Z-Image; licença comunitária própria |
| **Famegrid Natural V1** | LoKr/LoRA para Krea 2 | tira o aspecto excessivamente encenado/cinematográfico e força fotografia comum | carregado junto do Krea, peso padrão 0,7 e trigger `Famegrid` | incluído no tempo do Krea | não corrige sozinho mãos, olhos ou objetos incoerentes |
| **Mage-Flow-Edit-Turbo 4B** | editor por instrução | troca roupa, objetos ou fundo mantendo a foto como base | `foto editar` e `foto vestir` | **44–160 s** para a difusão, normalmente perto de 100 s | redesenha o quadro; o pipeline precisa recolocar a cabeça e casar o grão |
| **Qwen3-VL-4B** | encoder texto/visão | lê a instrução e as imagens para o Mage-Flow | automaticamente durante edição | incluído no tempo do Mage | não gera pixels sozinho |
| **FLUX.2 Klein 4B** | gerador/editor multi-reference | cria cenas novas com pessoa, roupa e objetos de várias referências; fallback de geração | `foto cena` ou quando Draw Things não está disponível | **212,6 s** para 3 referências em 896×1216 | identidade zero-shot é semelhante, não biométrica/pixel-idêntica |
| **Qwen 3 4B** | encoder texto/visão | interpreta prompt e referências do FLUX.2 | automaticamente em `foto cena`/FLUX.2 | incluído no tempo do FLUX.2 | não gera pixels sozinho |
| **RealVisXL V5 / SDXL** | checkpoint gerador | geração com LoRAs SDXL e polimento de microtextura com denoise 0,03 | `--motor sdxl`, `--lora` e estágio de polimento da edição | **48 s** para gerar 896×1152; **~132 s** no polimento medido | ecossistema excelente de LoRAs, mas aderência/anatomia abaixo dos modelos novos |
| **1x-ITF-SkinDiffDetail** | modelo 1× de detalhe | recupera microtextura de pele e tecido sem aumentar resolução | fim do polimento SDXL | incluído nos ~132 s | melhora textura; não conserta estrutura ou identidade |
| **SeedVR2** | upscaler generativo | amplia e reconstrói poros, fios, barba e tecido | `foto ampliar` ou `foto editar --ampliar` | **25,8 s** no teste 2× | pode inventar microdetalhes; em edição roda antes da cabeça original ser recolocada |
| **Vision.framework** | segmentação/detecção nativa | encontra rosto/cabelo e cria a máscara semântica do composite | estágio final de `foto editar` | aproximadamente **30–160 ms** | preservação estrita só funciona se pose e enquadramento não mudarem |

## Tempo ponta a ponta por caso de uso

| caso | rota completa | tempo típico/medido |
|---|---|---:|
| gerar rápido | Z-Image + Draw Things | **40–60 s** |
| gerar com máximo fotorrealismo | Krea 2 Q4 + Famegrid | **~3 min** em 384×512; **~11,5 min** em 640×896 |
| gerar com uma LoRA SDXL | RealVisXL + LoRA | **~48 s** no teste base; varia com steps/LoRA |
| editar mantendo rosto/cenário | Mage → RealVisXL 0,03 → cabeça original → grão | **~3–4 min** |
| editar e ampliar 2× | pipeline anterior + SeedVR2 antes do composite | aproximadamente **4 min** |
| criar cena com referências | FLUX.2 + Qwen encoder | **~3,5 min** no teste com 3 referências |
| ampliar uma imagem | SeedVR2 | **~26 s** no teste 2× |

O “tempo médio” acima deve ser lido como faixa operacional. Houve variação de
44 a 160 segundos no Mage para trabalhos semelhantes quando a memória estava
pressionada. O primeiro uso depois de iniciar um runtime também carrega os
pesos e pode demorar mais.

## Candidatos testados e removidos

| candidato | resultado nesta máquina | decisão |
|---|---|---|
| Qwen-Image-Edit-2511 Q4 | 341–712 s por imagem e mais drift global que Mage | removido do fluxo e do disco |
| Mage-Flow não-turbo | 1.767 s em 10 steps/CFG 2, sem melhoria visual | removido |
| Step1X-Edit | CUDA/Triton, sem execução MPS suportada e memória incompatível | não instalado |
| Z-Image BF16 no ComfyUI | duplicava o modelo i8x mais rápido do Draw Things | removido |
| 4x-UltraSharpV2 | 183 s no teste, menos eficiente que SeedVR2 e licença inadequada ao padrão público | removido |
| RealESRGAN x2/x4 | redundante diante de SeedVR2/Lanczos nas rotas atuais | removido |
| PuLID-Flux2 não oficial | integração/pesos não demonstraram uma carga de identidade válida | não usado |

## Regra prática

- Quer rapidez: **Z-Image/Draw Things**.
- Quer a fotografia mais natural possível e pode esperar: **Krea 2/Famegrid**.
- Quer alterar uma foto real: **Mage-Flow**, sempre pelo pipeline completo.
- Quer misturar pessoa, roupa e objetos de referências: **FLUX.2**, aceitando
  que o rosto não é uma cópia biométrica.
- Quer um estilo treinado específico: **SDXL + LoRA**.
- Quer apenas mais resolução: **SeedVR2**, conferindo os detalhes inventados.

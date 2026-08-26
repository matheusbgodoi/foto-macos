# Limites honestos

O que este pipeline **não** faz, e por quê. Escrito para poupar seu tempo.

---

## Não gera "você" numa cena nova

Você manda uma selfie e pede "eu sentado num café". A cena sai excelente, a
roupa sai certa, e **o rosto não é o seu** — é alguém parecido, com o cabelo
errado.

Isso não é ajuste de prompt nem de parâmetro. Modelos de edição por instrução
leem a referência como *descrição de pessoa*, não como identidade. Eles
recriam alguém compatível com a descrição.

### Por que não dá para resolver com um adapter

Adapters de identidade zero-shot — PuLID, InstantID, InfiniteYou, IP-Adapter
FaceID, PhotoMaker, USO, DreamO, UNO — foram feitos para **FLUX.1 e SDXL**, uma
geração atrás. Não existe nenhum para Mage-Flow, Qwen-Image-Edit-2511, Z-Image
ou FLUX.2 klein.

O único que **alega** suportar FLUX.2 klein (`Fayens/Pulid-Flux2` +
`iFayens/ComfyUI-PuLID-Flux2`) está **quebrado por construção**: os pesos trazem
84 tensores com prefixo `pulid_ca_double` / `pulid_ca_single` (dim 4096), o
código declara os módulos como `self.double_ca` / `self.single_ca`, e carrega com
`load_state_dict(state, strict=False)` sem nenhum remapeamento de chave. Toda a
camada de injeção de identidade é **descartada em silêncio** e fica com
inicialização aleatória. No klein 4B (hidden 3072 ≠ 4096) ele ainda constrói um
injetor novo com `nn.init.normal_(std=0.02)`.

Se você usou isso e viu membros a mais e roupa derretida: é o resultado
matemático esperado de somar ruído gaussiano no residual de cada bloco.

### Rotas que restam

| rota | veredito |
|---|---|
| **Editar uma foto sua existente** | é o que este pipeline faz, e funciona |
| **LoRA treinada no rosto** | único método confiável — mas precisa de 20–30 fotos, e treinar difusão em MPS tem bug de **falha silenciosa** em todas as rotas testadas (gasta horas e entrega um arquivo que carrega e não faz nada) |
| **API fechada** | quando identidade importa mais que rodar local |

---

## Inventa acessórios na área que redesenha

Medido: ao trocar uma roupa, o modelo moveu o relógio para o pulso errado,
adicionou um cinto que não existia, e numa tentativa colocou relógio nos **dois**
pulsos.

**Prompt negativo não resolve.** Com CFG 1 — regime distilled/turbo, que é o que
torna 4 steps viáveis — o negativo é essencialmente ignorado. Pagar CFG real
dobra o custo por step e, medido aqui, **não melhora a qualidade**.

O que funciona hoje: descrever explicitamente o que **preservar** no prompt
positivo, e uma segunda passada dedicada a remover o que apareceu de errado.

---

## Não é indistinguível sob inspeção próxima

O estágio de polimento aproxima muito a microtextura da original (5,71 → 4,73 →
**5,61** de ruído de alta frequência na região editada). Mas a peça gerada nunca
tem a *história* de uma peça real: a trama irregular, o amassado assimétrico, o
fiapo.

Num relance, ou numa tela de celular, passa. Ampliado na peça editada,
distingue-se.

---

## Perde do ChatGPT em algumas dimensões

Sem torcida:

| dimensão | quem ganha |
|---|---|
| preservar o que não foi editado | **local** — nenhuma ferramenta web dá esse controle |
| realismo de pele | empate ou **local** (o GPT-Image é plastificado) |
| aderência a prompt complexo | **GPT-Image**, com folga |
| texto dentro da imagem | **GPT-Image** |
| velocidade | **GPT-Image** — ~1 s numa GPU dedicada contra ~2 min aqui |

---

## Restrições de hardware

24 GB de memória unificada dão ~16–18 GB úteis. Consequências práticas:

- o mesmo trabalho varia de **44 s a 156 s** conforme a pressão de memória;
- com outra coisa pesada rodando, o tempo **triplica** (um upscale de 183 s
  chegou a 909 s competindo com um download);
- o primeiro uso após ligar o ComfyUI paga ~40 s a mais para carregar o modelo;
- modelos de 20B só cabem quantizados, e aí ficam lentos demais para iterar.

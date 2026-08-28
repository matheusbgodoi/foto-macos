# Checkpoint — 27 de agosto de 2026

Este documento congela o estado da investigação antes da pausa solicitada pelo
Matheus. Nenhum dos resultados abaixo deve ser promovido silenciosamente como
"final": a identidade do último composite ficou boa, mas a cena restante foi
reprovada por aparência artificial.

## Veredito visual atual

- A máscara semântica + composição multibanda consegue devolver o rosto real do
  Matheus sem a antiga borda oval. No último teste, a identidade ficou boa.
- O restante da imagem — pessoas ao fundo, olhos, mãos, objetos, iluminação e
  microtextura — continua artificial. O resultado completo está **reprovado**.
- Entre as gerações somente com a LoRA, o melhor ponto continua sendo o
  checkpoint `step750` com peso `1.1`. Peso `1.4` deforma; trocar seed e prompt
  ainda muda demais a identidade.
- O `step750 w1.1` prova que a LoRA aprendeu características do rosto, mas não
  que o Krea 2 Turbo consiga manter simultaneamente identidade, anatomia e
  realismo de toda a cena em qualquer composição.
- ArcFace é apenas uma métrica auxiliar. A avaliação visual do usuário tem
  prioridade: uma pontuação alta não compensa olhos ruins, pele artificial ou
  uma cena incoerente.

## Ranking de identidade aprovado pelo usuário

1. `eval/matheus-musubi-step750-w11.png`
2. `eval/matheus-musubi-step750-s20260828-w11.png`
3. a referência real frontal enviada pelo usuário

As demais variantes não devem ser usadas como demonstração de qualidade.

## Artefatos avaliados

Os arquivos ficam em `~/Downloads` e **não** fazem parte do repositório.

| arquivo | execução / métrica | veredito |
|---|---|---|
| `matheus_faculdade_macbook_base.png` | Krea, 768×1024, 174,5 s, ArcFace 0,5735 | reprovado |
| `matheus_faculdade_macbook_base_v2.png` | Krea, 640×896, 127,2 s, ArcFace 0,5778 | reprovado |
| `matheus_faculdade_macbook_polido.png` | polimento em 64,1 s, ArcFace 0,6138 | melhor base Krea, ainda insuficiente visualmente |
| `matheus_faculdade_macbook_seedvr2_s0.png` | SeedVR2 2× em 88,5 s | suavizou/inventou detalhes; reprovado |
| `matheus_faculdade_identidade_famegrid03.png` | LoRA de identidade 1,1 + Famegrid 0,3, 316,8 s, ArcFace 0,5972 | cena mais variada, rosto ainda plástico e custo alto |
| `matheus_faculdade_macbook_ultrarealista_2x.png` | híbrido, 1296×1816, ArcFace 0,6148, QA de costura 0,84 | usuário identificou artefatos; reprovado |
| `matheus_faculdade_real_final.png` | base real + edição + composite semântico, QA de costura 0,86 | rosto parece o Matheus; resto artificial; reprovado |

### Melhor matéria-prima para a próxima tentativa

`data/016.jpg` é uma foto real do Matheus já sentado diante de um notebook em
ambiente universitário. Ela é uma base melhor do que gerar o corpo inteiro de
novo: permite preservar rosto, cabelo, mãos, notebook e mesa como pixels reais.

## Incidentes e correções concluídas

### Quadro preto do Mage-Flow

A primeira tentativa de editar a foto real produziu um PNG preto. O arquivo de
entrada estava normal; o log mostrou `RuntimeWarning: invalid value encountered
in cast`. O autostart do ComfyUI ainda usava `--force-fp16`, incompatível com o
Mage-Flow no MPS, embora a documentação já alertasse sobre isso.

Correções:

- removido `--force-fp16` do serviço;
- `--reserve-vram` reduzido de 6 para 2 GB;
- o pipeline agora rejeita arquivos ausentes, pretos ou sem variação antes do
  polimento SDXL.

Depois da reinicialização sem FP16 forçado, o Mage produziu
`out/faculdade_real_bf16_00001_.png`, válido, em 56,3 s a 768×1360. A edição
incluiu moletom, fones, professor e alunos, mas re-renderizou o rosto e deixou
os elementos de fundo artificiais.

### Polimento extremamente lento

`matheus_faculdade_real_polido.png` levou 3581,8 s. O Mac ficou com a tampa
fechada durante o trajeto de metrô, portanto esse número **não é benchmark**.
Na retomada, o teste deve ser repetido com o Mac na tomada, tampa aberta e sem
outros modelos ocupando memória. Também será necessário descarregar o Mage
antes de carregar o SDXL.

### Caminho absoluto no polimento

O `SaveImage` do ComfyUI não aceita salvar diretamente fora de sua pasta de
output. `polir.py` agora usa um prefixo temporário interno e copia o resultado
para o destino absoluto pedido pelo CLI/MCP.

### Upscale e identidade

SeedVR2 continua generativo mesmo com `softness=0`; pode suavizar pele e alterar
olhos. O CLI/MCP agora expõe três modos:

- `fiel`: softness 0, padrão para pessoas;
- `equilibrado`: softness 0,5;
- `criativo`: softness 0,75, maior risco de alterar traços.

Nenhum upscale generativo deve tocar o rosto final sem comparação visual.

## Estado da pausa

- ComfyUI: **desligado**; `127.0.0.1:8188` não responde.
- Geração/edição/treino no Mac: **nenhum processo ativo**.
- MCPs e configurações dos conectores permanecem instalados, mas ficam ociosos
  enquanto o ComfyUI está desligado.
- RTX 3090 remota: não foi acionada nem alterada neste fechamento.
- LoRA Krea `step750 w1.1`: preservada como melhor candidata experimental; não
  apagar nem substituir até o próximo comparativo controlado.

## Próximos passos — ordem de retomada

Nada desta seção deve rodar até o usuário pedir para continuar.

1. **Estabilizar memória e medir tempo real.** Implementar descarregamento
   explícito (`POST /free`) entre Mage e SDXL; repetir somente um benchmark com
   tampa aberta, Mac na tomada e memória limpa.
2. **Trocar a arquitetura da imagem.** Parar de pedir ao Krea + LoRA que gere a
   pessoa e a sala inteiras. Começar em `data/016.jpg` e preservar como pixels
   reais: cabeça, cabelo, mãos, notebook, mesa e corpo sempre que possível.
3. **Gerar somente o fundo.** Usar Krea 2 + Famegrid sem LoRA de identidade para
   criar a sala de aula; manter pessoas distantes, de costas ou desfocadas, sem
   texto legível, reduzindo olhos e letras defeituosos.
4. **Editar por máscaras semânticas.** Alterar apenas moletom e fones em máscaras
   específicas. Não re-renderizar o sujeito inteiro. Compor fundo e roupa na
   grade nativa usando a mistura multibanda já validada.
5. **Aplicar textura somente onde foi gerado.** Proteger rosto e mãos. Usar
   SeedVR2 `fiel` apenas em fundo/roupa; se houver drift, usar Lanczos no sujeito
   e SeedVR2 somente nas regiões sintéticas.
6. **Criar portas de qualidade.** Reprovar automaticamente quadro preto e texto
   ilegível; detectar faces pequenas no fundo; revisar olhos, número/formato de
   mãos e identidade. ArcFace continua consultivo, não decisivo.
7. **Comparar uma alternativa de identidade.** Depois de finalizar a rota
   híbrida, testar na RTX 3090 uma LoRA de pessoa em FLUX.1 Dev, cujo ecossistema
   de treino de identidade é mais maduro, usando o mesmo dataset, prompts e
   seeds. Manter a LoRA Krea arquivada para comparação; não retreinar agora.
8. **Só então gerar o candidato final**, salvar em `~/Downloads` e pedir a
   aprovação visual antes de promover qualquer preset ou documentação de
   produção.

### Decisão incorporada da conversa lateral

Krea 2 não será apagado ou substituído agora. A combinação de adapters já foi
executada de verdade: identidade `step750` em 1,1 + Famegrid em 0,3. Ela melhora
a estética geral, mas não atingiu o realismo da imagem de evento nem tornou a
identidade robusta.

A decomposição em duas passagens continua sendo uma boa hipótese, com uma
restrição técnica importante: o Krea 2 Q4 usado no Mac pelo MFLUX é atualmente
texto-para-imagem. Ele não oferece uma segunda passagem img2img/inpaint
mascarada para aplicar a LoRA apenas no rosto. Portanto:

- **comprovado:** as duas LoRAs carregam juntas numa geração completa;
- **não comprovado:** Famegrid gerar a cena e uma segunda passagem Krea alterar
  somente rosto/cabelo sem costura ou drift;
- **onde validar:** ComfyUI/CUDA na RTX 3090, se houver uma rota Krea mascarada
  funcional, comparando contra Krea em passagem única e FLUX.1 Dev;
- **fallback mais seguro no Mac:** começar de uma foto real e preservar pixels
  do sujeito, gerando somente fundo/roupa por máscaras semânticas.

Não fundir identidade e Famegrid em um único adapter antes desses testes. Manter
adapters separados conserva controle de peso, permite A/B e evita decorar um
único estilo fotográfico dentro da identidade.

## Critério de sucesso na retomada

O resultado só passa se cumprir simultaneamente:

- é reconhecivelmente o Matheus, sem depender de capuz, sombra ou seed favorável;
- rosto, cabelo, olhos e mãos não mostram reconstrução artificial;
- pessoas ao fundo não têm olhos/membros anômalos;
- a textura da roupa e da sala é fotográfica, sem pele plástica;
- não há texto/logos inventados;
- a costura do composite não é perceptível em tamanho normal nem no realce.

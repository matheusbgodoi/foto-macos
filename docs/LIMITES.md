# Limites honestos

## Editar não é o mesmo que recriar

`foto editar` e `foto vestir` trabalham sobre a fotografia original. Quando a
cabeça fica na mesma posição, o pipeline recoloca os pixels reais por
segmentação do Vision.framework e blend multi-banda. Isso dá preservação muito
mais forte que pedir ao modelo para “não mudar o rosto”.

Esse método não se aplica quando pose, ângulo ou enquadramento da cabeça mudam.
Nesses casos use `foto cena` e aceite a natureza zero-shot da identidade.

## Identidade em uma cena nova

FLUX.2 Klein aceita várias referências e, no teste com rosto + corpo + roupa,
produziu uma pessoa coerente, com roupa e anatomia corretas. Ainda assim, o
resultado é uma síntese das referências, não uma cópia biométrica garantida.
Compare rosto, cabelo, óculos, mãos e acessórios antes de publicar.

Não há promessa de “pixel idêntico” quando a cena é criada do zero: não existem
pixels de origem na nova pose para preservar.

## Anomalias locais

Qualquer modelo pode inventar relógio, cinto, dedos, texto ou detalhes da roupa.
Prompt negativo quase não ajuda nos modelos distilled com CFG 1. A abordagem
mais segura é:

1. pedir explicitamente o que deve permanecer;
2. gerar rascunho em ~1 MP;
3. inspecionar rosto, mãos, pulsos, acessórios e texto;
4. fazer uma edição localizada se necessário;
5. ampliar somente o resultado aprovado.

## Upscale não transforma uma geração ruim em foto real

SeedVR2 adiciona microtextura plausível, mas é generativo e pode alterar traços.
No fluxo de edição ele toca somente a imagem editada e a cabeça original é
recolocada depois. Para preservar tudo estritamente, não use `--ampliar`; use o
arquivo na resolução nativa ou Lanczos.

Em fontes pequenas (por exemplo, JPEG de WhatsApp com 640 px), ampliar o quadro
generativamente e a cabeça por Lanczos cria diferença visível de nitidez. O
resultado recomendado é o nativo; resolução perdida na origem não pode ser
recuperada sem alguma invenção.

## Estilos

Os presets resolvem estilos amplos por prompt. Estilos muito específicos — um
artista, personagem, produto ou linguagem visual consistente — normalmente
exigem uma LoRA/checkpoint SDXL compatível. O roteador muda para SDXL quando
`--lora` é fornecido.

## Desempenho

24 GB de memória unificada são suficientes para os modelos 4B escolhidos, mas
pressão de memória causa grande variação. Evite rodar outros modelos ou tarefas
pesadas em paralelo. O primeiro uso também paga o carregamento dos pesos.

Draw Things não foi desinstalado porque o runtime Metal quantizado i8x é mais
rápido e econômico para geração. ComfyUI continua indispensável para edição,
workflows visuais e multi-referência. O sistema unifica a interface, não força
um único runtime a ser pior em todas as tarefas.

## Privacidade e segurança

O fluxo é local. Mesmo assim, imagens sintéticas de pessoas podem ser usadas de
forma enganosa. Obtenha consentimento de terceiros e não trate semelhança
visual como autenticação de identidade.

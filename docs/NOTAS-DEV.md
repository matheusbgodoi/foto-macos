# Notas de desenvolvimento

## Invariantes do fluxo

1. Mage-Flow Turbo roda em 4 steps/CFG 1. O não-turbo não trouxe ganho medido.
2. Todo passe generativo acontece antes do composite da cabeça.
3. A foto original nunca passa pelo SeedVR2.
4. A cabeça é segmentada pelo Vision.framework, harmonizada em tom e composta
   por pirâmide Laplaciana com feather base de 2 px.
5. `TextEncodeMageFlowEdit` recebe referências da API como
   `images.image_1`, `images.image_2`, etc.; sem o prefixo o nó ignora a imagem.
6. `foto vestir` aceita descrição textual. Uma foto de outra pessoa vestindo a
   peça não entra silenciosamente no editor; para isso existe `foto cena`.

## Roteamento

- `gerar_coringa.py`: Draw Things/Z-Image i8x quando instalado; FLUX.2 como
  fallback; SDXL ao receber LoRA.
- `flux2.py`: text-to-image e nova cena multi-referência.
- `pipeline.py`: edição sobre foto existente.
- `ampliar.py`: SeedVR2/MLX; fallback Lanczos.

## Medições que não devem ser esquecidas

| teste | resultado |
|---|---|
| Mage Turbo, edição ~1 MP | 44–160 s |
| Mage não-turbo, 10 steps/CFG 2 | 1.767 s, sem ganho |
| Qwen-Image-Edit-2511 Q4 | 341–712 s, mais drift |
| FLUX.2, 3 refs, 896×1216 | 212,6 s |
| Z-Image i8x/Draw Things, preset iPhone | 54,7 s |
| SeedVR2 2× | 25,8 s |

## Workflows visuais

`sync_workflows.py` deriva grafos dos templates oficiais da versão instalada do
ComfyUI e grava a mesma cópia em `workflows/comfyui/` e
`~/comfyui/user/default/workflows/foto-macos/`. Não edite o arquivo gerado;
duplique-o no ComfyUI para experimentos.

## Código legado removido

O face-swap afim, o composite por diferença de pixels e o UltraSharp não fazem
parte do vencedor. Eles foram removidos porque deixavam costura, otimizavam uma
métrica errada ou tinham licença inadequada para o instalador de produção.

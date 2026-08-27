# Workflows visuais

Estes arquivos sao gerados por `src/sync_workflows.py` a partir dos templates
oficiais da versao instalada do ComfyUI. Eles aparecem em **Workflows > Browse**
na pasta `foto-macos` depois de recarregar `http://127.0.0.1:8188`.

1. **Editar foto / Mage** — mesmo motor do `foto editar`, antes do polimento e
   do composite externo.
2. **Referencias e cena / FLUX.2** — pessoa, roupa e estilo como referencias;
   candidato para `foto cena`.
3. **Gerar versatil / FLUX.2** — texto para imagem local no ComfyUI.
4. **Gerar rapido / Z-Image** — chama o mesmo Draw Things i8x do CLI/MCP; nao
   mantem uma copia BF16 mais lenta dentro do ComfyUI. O no visual roteia
   apenas runtimes externos Apple; SDXL/FLUX usam os grafos nativos 1--3 para
   nao chamar a fila do ComfyUI de dentro dela mesma.
5. **Fotorrealismo / Krea 2 + Famegrid** — chama o mesmo runtime MLX Q4 do
   CLI/MCP por um no customizado; nao duplica o checkpoint dentro do ComfyUI.
6. **Identidade por referencia / Krea 2 ReID** — workflow oficial de identidade
   zero-shot com um retrato; usa Krea 2 INT8 ConvRot e roda diretamente no
   ComfyUI. O grafo foi validado em CUDA; no Apple MPS, o INT8 ConvRot exige
   `aten::_int_mm`, cai no CPU e nao e praticavel. Fica instalado para abrir,
   estudar e usar em um host NVIDIA, mas nao entra no roteador do Mac.

O pipeline completo de preservacao de identidade tambem usa Vision.framework
e SeedVR2/MLX. Essas etapas nao sao nos do ComfyUI; o grafo visual mostra o
nucleo de difusao, enquanto o CLI/MCP orquestra o acabamento externo.

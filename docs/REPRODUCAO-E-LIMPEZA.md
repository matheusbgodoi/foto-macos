# Reprodução, retenção e limpeza segura

Este documento define como liberar espaço no Mac e na RTX 3090 sem perder a
capacidade de reproduzir o pipeline. Limpeza é uma etapa final, depois da
aprovação visual e de um inventário em modo simulação — nunca durante uma rodada
experimental.

## Regra central

O GitHub público guarda a **receita**, não dados pessoais nem pesos sem direito
de redistribuição. Reproduzir exatamente uma LoRA de identidade exige também um
pequeno pacote privado de recuperação.

### GitHub público

Deve conter:

- código dos runners, MCP, roteador e workflows do ComfyUI;
- versões/revisões dos repositórios e dependências;
- URLs oficiais, versões, licenças e SHA-256 dos modelos;
- configurações completas de treino e inferência;
- prompts, seeds, resoluções, samplers, pesos de LoRA e tempos;
- critérios de seleção/rejeição e resultados numéricos;
- bugs, patches e comandos de instalação;
- manifesto com nomes lógicos dos artefatos privados, sem fotos ou tokens.

### Pacote privado de recuperação

Manter fora do Git e, de preferência, em armazenamento criptografado:

- a LoRA vencedora `step750` e seu SHA-256;
- manifesto/UUIDs das 24 fotos selecionadas no Apple Photos;
- captions finais e a lista de fotos rejeitadas;
- checkpoint(s) adicionais somente enquanto o comparativo não terminar;
- arquivo privado `identities.json`, sem credenciais de serviços;
- versões exatas do driver NVIDIA, CUDA, PyTorch, Musubi e MFLUX;
- logs curtos da execução vencedora e a matriz de avaliação por prompt/seed.

Se a biblioteca Fotos e os UUIDs forem preservados, as imagens podem ser
reexportadas. Para recuperação realmente exata, mantenha também uma cópia
criptografada do dataset selecionado. Apagar dataset, UUIDs e LoRA de todas as
máquinas torna impossível prometer reprodução bit a bit apenas com o GitHub.

## Estado que deve ser preservado até a decisão final

- Krea 2 Turbo Q4 e Famegrid no Mac;
- LoRA Krea Musubi `step750`, peso de referência 1,1;
- checkpoints Krea ainda necessários para o A/B pendente;
- dataset selecionado e captions na RTX 3090;
- FLUX.1 Dev e ambiente de treino apenas se forem necessários ao comparativo;
- modelos ativos do editor Mage, FLUX.2, RealVisXL e SeedVR2;
- todos os workflows versionados em `workflows/comfyui`.

Nada desta lista deve ser removido durante a pausa atual.

## Candidatos a remoção depois da aprovação

### Mac

- caches de downloads incompletos e duplicados;
- variantes Krea Q8/BF16 derrotadas, mantendo somente Q4 de produção;
- checkpoints ReID CUDA que não rodam em MPS;
- Qwen-Image-Edit e Step1X quando não houver rota ativa;
- outputs temporários e comparativos reprovados;
- checkpoints ou upscalers substituídos, depois de registrar URL, revisão,
  licença e hash.

O arquivo [MODELOS.md](MODELOS.md) já registra vários descartes medidos. Antes
de remover qualquer novo peso, atualizar aquela tabela.

### RTX 3090

- caches de latentes/text encoder depois de encerrar todos os treinos;
- estados do otimizador e checkpoints intermediários derrotados;
- clones/venvs duplicados de AI Toolkit, SimpleTuner ou Musubi;
- cópias duplicadas de Krea Raw/Turbo, VAE e encoder;
- outputs LoKr comprovadamente inválidos;
- FLUX.1 Dev e seus caches somente depois de decidir se ele perde o comparativo.

A LoRA vencedora e o pacote privado de recuperação devem sair do disco de
trabalho apenas depois de existirem em pelo menos um backup verificado.

## Procedimento obrigatório de limpeza

1. Parar ComfyUI, MLX, treino e qualquer processo que mantenha pesos abertos.
2. Inventariar Mac e RTX 3090 com caminho absoluto, tamanho, mtime e SHA-256.
3. Classificar cada item como `produção`, `recuperação`, `temporário` ou
   `candidato a apagar`.
4. Registrar no GitHub a origem, revisão, licença, configuração e hash de tudo
   que será removido.
5. Criar/verificar o pacote privado de recuperação.
6. Gerar uma lista de remoção em **dry-run** e pedir aprovação explícita do
   usuário.
7. Remover somente caminhos explícitos; nunca usar glob amplo, `~` ou diretório
   raiz como alvo.
8. Executar `foto status`, os testes e uma geração curta após a limpeza.
9. Registrar espaço liberado e itens removidos.

## Receita mínima do treino Krea atual

- base de treino: Krea 2 Raw;
- inferência: Krea 2 Turbo Q4 no MFLUX;
- dataset: 24 fotos selecionadas de 2025–2026;
- resolução: 768;
- treinador: Musubi Tuner;
- rede: `networks.lora_krea2`, dim/alpha 32/32;
- precisão: base FP8 escalada, adapter BF16;
- optimizer: AdamW 8-bit, learning rate `1e-4`;
- timestep: `krea2_shift`;
- troca de blocos: 16;
- checkpoints: a cada 250 passos, até 2000;
- vencedor visual atual: step 750, peso 1,1;
- Famegrid: desligado para o baseline de identidade; 0,3 no A/B combinado;
- peso 1,4: reprovado.

O comando executável está em `scripts/train-krea2-identity.sh`; a análise
completa está em [IDENTIDADE.md](IDENTIDADE.md).

## Experimentos ainda necessários antes de limpar

1. Krea step750 em 1,0–1,1, com e sem Famegrid 0,2–0,4, nas mesmas seeds.
2. Verificar na RTX 3090 se existe uma segunda passagem Krea mascarada realmente
   funcional; o MFLUX do Mac não oferece essa operação.
3. Comparar FLUX.1 Dev LoRA no mesmo dataset/prompts/seeds.
4. Validar a rota híbrida de foto real + fundo/roupa gerados por máscara.
5. Escolher o vencedor por conjunto de imagens, não pela melhor seed isolada.

Somente depois desses cinco pontos será possível determinar o conjunto mínimo
de modelos de produção e apagar o restante com segurança.

## Segredos

Tokens do CivitAI, Hugging Face, GitHub e SSH nunca entram em arquivos do
repositório, logs publicados ou exemplos de configuração. Uma chave exposta em
chat deve ser revogada e substituída; o novo valor deve ficar apenas no keychain
ou em arquivo privado com permissão restrita.

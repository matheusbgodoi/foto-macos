/** Pi extension for foto-macos.
 *
 * Pi does not ship an MCP client. This native adapter exposes the same Python
 * entry points used by the MCP server, so there is still one image pipeline.
 */
import { spawn } from "node:child_process";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { Type } from "typebox";

interface ToolResult { content: Array<{ type: string; text?: string }>; details?: unknown }
interface PiApi {
  registerTool(tool: {
    name: string; label: string; description: string; parameters: unknown;
    execute: (
      id: string, params: Record<string, unknown>, signal?: AbortSignal,
      onUpdate?: (update: ToolResult) => void,
    ) => Promise<ToolResult>;
  }): void;
}

const PY = process.env.FOTO_PYTHON || join(homedir(), "comfyui", ".venv", "bin", "python");
const ROOT = process.env.FOTO_MACOS_ROOT
  || resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const script = (name: string) => `${ROOT}/src/${name}`;
const text = (value: string): ToolResult => ({ content: [{ type: "text", text: value }] });

function run(
  name: string, args: string[], signal?: AbortSignal,
  onUpdate?: (update: ToolResult) => void,
): Promise<string> {
  return new Promise((resolve, reject) => {
    const child = spawn(PY, [script(name), ...args], { env: process.env });
    let output = "";
    const receive = (chunk: Buffer) => {
      output += chunk.toString();
      onUpdate?.(text(output.slice(-1000)));
    };
    child.stdout.on("data", receive);
    child.stderr.on("data", receive);
    const stop = () => child.kill("SIGTERM");
    signal?.addEventListener("abort", stop, { once: true });
    child.on("error", reject);
    child.on("close", (code) => {
      signal?.removeEventListener("abort", stop);
      if (code === 0) resolve(output.trim());
      else reject(new Error(output.slice(-2000) || `processo encerrou com codigo ${code}`));
    });
  });
}

export default function fotoMacos(pi: PiApi): void {
  const generate = async (
    _id: string, p: Record<string, unknown>, signal?: AbortSignal,
    update?: (result: ToolResult) => void,
  ) => {
    const args = ["gerar", String(p.prompt)];
    if (p.saida) args.push("--saida", String(p.saida));
    if (p.tamanho) args.push("--tamanho", String(p.tamanho));
    if (p.estilo) args.push("--estilo", String(p.estilo));
    if (p.motor) args.push("--motor", String(p.motor));
    if (p.seed) args.push("--seed", String(p.seed));
    return text(await run("foto.py", args, signal, update));
  };
  const generateSchema = Type.Object({
    prompt: Type.String({ description: "Imagem a criar, em portugues ou ingles." }),
    saida: Type.Optional(Type.String({ description: "Caminho absoluto do PNG final." })),
    tamanho: Type.Optional(Type.String({ description: "LARGURAxALTURA; padrao 1024x1024." })),
    estilo: Type.Optional(Type.String({
      description: "auto, foto-natural, iphone, profissional, produto, cartoon, pixel-art, ilustracao, anime ou famegrid",
    })),
    motor: Type.Optional(Type.String({ description: "auto, drawthings, krea2, sdxl ou flux2" })),
    seed: Type.Optional(Type.Number()),
  });

  pi.registerTool({
    name: "foto_gerar", label: "Foto · gerar",
    description: "Gera imagem localmente e roteia entre Z-Image, Krea 2/Famegrid, SDXL e FLUX.2.",
    parameters: generateSchema, execute: generate,
  });
  // Compatibilidade com conversas antigas do Pi; aponta para o roteador novo.
  pi.registerTool({
    name: "image_generate", label: "Foto · gerar",
    description: "Alias de foto_gerar; gera imagem pelo pipeline foto-macos.",
    parameters: generateSchema, execute: generate,
  });

  pi.registerTool({
    name: "foto_editar", label: "Foto · editar",
    description: "Edita uma foto existente por instrucao, preservando rosto/cabelo originais.",
    parameters: Type.Object({
      foto: Type.String(), instrucao: Type.String(),
      saida: Type.Optional(Type.String()), ampliar: Type.Optional(Type.Boolean()),
      seed: Type.Optional(Type.Number()),
    }),
    execute: async (_id, p, signal, update) => {
      const args = ["editar", String(p.foto), String(p.instrucao)];
      if (p.saida) args.push("--saida", String(p.saida));
      if (p.ampliar) args.push("--ampliar");
      if (p.seed) args.push("--seed", String(p.seed));
      return text(await run("foto.py", args, signal, update));
    },
  });

  pi.registerTool({
    name: "foto_cena", label: "Foto · referencias",
    description: "Compoe cena nova usando 1-4 imagens de referencia via FLUX.2.",
    parameters: Type.Object({
      prompt: Type.String(), referencias: Type.Array(Type.String()),
      saida: Type.Optional(Type.String()), tamanho: Type.Optional(Type.String()),
      seed: Type.Optional(Type.Number()),
    }),
    execute: async (_id, p, signal, update) => {
      const args = ["cena", String(p.prompt)];
      for (const ref of p.referencias as string[]) args.push("--ref", ref);
      if (p.saida) args.push("--saida", String(p.saida));
      if (p.tamanho) args.push("--tamanho", String(p.tamanho));
      if (p.seed) args.push("--seed", String(p.seed));
      return text(await run("foto.py", args, signal, update));
    },
  });

  pi.registerTool({
    name: "foto_ampliar", label: "Foto · ampliar",
    description: "Amplia uma imagem com SeedVR2 via MLX.",
    parameters: Type.Object({
      imagem: Type.String(), saida: Type.Optional(Type.String()),
      escala: Type.Optional(Type.Number()),
    }),
    execute: async (_id, p, signal, update) => {
      const args = ["ampliar", String(p.imagem)];
      if (p.saida) args.push("--out", String(p.saida));
      if (p.escala) args.push("--escala", String(p.escala));
      return text(await run("foto.py", args, signal, update));
    },
  });

  pi.registerTool({
    name: "civitai_modelo", label: "Civitai · inspecionar",
    description: "Consulta tipo, base, arquivos e hashes de um modelo Civitai. O token fica no Keychain.",
    parameters: Type.Object({ referencia: Type.String() }),
    execute: async (_id, p, signal, update) =>
      text(await run("civitai.py", ["info", String(p.referencia)], signal, update)),
  });

  pi.registerTool({
    name: "civitai_baixar", label: "Civitai · baixar",
    description: "Baixa um recurso Civitai e valida SHA-256; confira a licenca antes do uso comercial.",
    parameters: Type.Object({
      referencia: Type.String(), destino: Type.Optional(Type.String()),
      arquivoId: Type.Optional(Type.Number()),
    }),
    execute: async (_id, p, signal, update) => {
      const args = ["baixar", String(p.referencia)];
      if (p.destino) args.push("--destino", String(p.destino));
      if (p.arquivoId) args.push("--arquivo", String(p.arquivoId));
      return text(await run("civitai.py", args, signal, update));
    },
  });
}

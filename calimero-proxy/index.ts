import express, { Request, Response } from 'express';
import fs from 'fs';
import path from 'path';
import bodyParser from 'body-parser';
import { fileURLToPath } from 'url';
import type { CalimeroApp, Context } from '@calimero-network/calimero-client';
import { AbiClient } from './api/AbiClient';

// __dirname replacement for ESM
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Environment configuration (accept CALIMERO_*, NEXT_PUBLIC_*, and VITE_* vars)
const PORT: number = process.env.PORT ? Number(process.env.PORT) : 2137;
const API_BASE_URL: string =
  process.env.CALIMERO_API_URL ||
  process.env.NEXT_PUBLIC_API_URL ||
  process.env.VITE_NODE_URL ||
  'http://localhost:2428';
const JSONRPC_PATH: string =
  process.env.CALIMERO_JSONRPC_PATH || process.env.VITE_RPC_PATH || '/jsonrpc';
const APPLICATION_ID: string =
  process.env.CALIMERO_APPLICATION_ID ||
  process.env.NEXT_PUBLIC_APPLICATION_ID ||
  (process.env.VITE_APPLICATION_ID as string) ||
  (process.env.VITE_CONTEXT_ID as string) ||
  '4bZJB5vmPAPn7yYwPDwDfJmpEbTKdFm3pvcmSpZMDuYx';

if (!APPLICATION_ID) {
  // eslint-disable-next-line no-console
  console.warn(
    'Warning: Application/Context ID not set (expected one of CALIMERO_APPLICATION_ID, NEXT_PUBLIC_APPLICATION_ID, VITE_APPLICATION_ID, or VITE_CONTEXT_ID). RPC calls will fail until it is provided.',
  );
}

const app = express();
app.use(bodyParser.json({ limit: '100mb' }));

// Helper to read and encode the model file
function readModelFileBase64(): string {
  const modelPath = path.resolve(__dirname, '../trained-model.keras');
  if (!fs.existsSync(modelPath)) {
    const err: NodeJS.ErrnoException = new Error(
      `Model file not found at ${modelPath}`,
    ) as NodeJS.ErrnoException;
    // @ts-ignore custom code
    err.code = 'MODEL_NOT_FOUND';
    throw err;
  }
  const bytes = fs.readFileSync(modelPath);
  return bytes.toString('base64');
}


// Minimal JSON-RPC executor (no Calimero SDK)
type ExecuteParams<Args> = {
  applicationId: string;
  method: string;
  argsJson: Args;
};

async function executeJsonRpc<Args, Output>(params: ExecuteParams<Args>): Promise<{ ok: boolean; result?: Output; error?: string }> {
  const url = `${API_BASE_URL}${JSONRPC_PATH}`;
  const payload = {
    jsonrpc: '2.0',
    id: Date.now(),
    method: 'execute',
    params,
  };
  try {
    const resp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!resp.ok) {
      return { ok: false, error: `HTTP ${resp.status}` };
    }
    const data = await resp.json();
    if (data?.error) {
      const msg = data.error?.data?.cause?.info?.message || data.error?.message || 'RPC error';
      return { ok: false, error: String(msg) };
    }
    const inner = data?.result?.result ?? data?.result;
    return { ok: true, result: inner as Output };
  } catch (e: any) {
    return { ok: false, error: e?.message || 'Network error' };
  }
}

// CalimeroApp adapter backed by minimal executor (no sdkApp)
const rpcApp = {
  async execute<Args, Output>(context: Context, method: string, args: Args) {
    const appId = (context as any)?.applicationId || (context as any)?.id || APPLICATION_ID;
    const { ok, result, error } = await executeJsonRpc<Args, Output>({
      applicationId: appId,
      method,
      argsJson: args as any,
    });
    if (!ok) {
      return { success: false, error: error || 'Execution failed' };
    }
    return { success: true, result: result as Output };
  },
};

// Build AbiClient using our minimal adapter and a lightweight Context
function buildAbiClient(): AbiClient {
  const context = ({ applicationId: APPLICATION_ID } as unknown) as Context;
  return new AbiClient(rpcApp as CalimeroApp, context);
}

// Single POST endpoint to upload the current model
app.post('/upload-model', async (req: Request, res: Response) => {
  try {
    // Read and encode file
    const fileBase64 = readModelFileBase64();

    // Prepare parameters according to abi.json (upload_current_model)
    const nowMs = Date.now();
    const {
      name = 'trained-model',
      description = 'Uploaded via calimero-proxy',
      model_type = 'keras',
      version = '1.0.0',
      uploader = process.env.MODEL_UPLOADER || 'server',
      prediction_accuracy = 0.0,
      date = nowMs,
      model_params = '',
      is_public = true,
    } = (req.body || {}) as Record<string, unknown>;

    const args = {
      name: String(name),
      description: String(description),
      model_type: String(model_type),
      version: String(version),
      file_bytes_base64: String(fileBase64),
      uploader: String(uploader),
      prediction_accuracy: Number(prediction_accuracy),
      date: typeof date === 'number' ? (date as number) : nowMs,
      model_params: String(model_params),
      is_public: Boolean(is_public),
    } as const;

    const abi = buildAbiClient();
    const result = await abi.uploadCurrentModel(args as any);

    return res.status(200).json({ ok: true, result: result ?? null });
  } catch (error: any) {
    if (error?.code === 'MODEL_NOT_FOUND') {
      return res.status(404).json({ ok: false, error: error.message });
    }
    // eslint-disable-next-line no-console
    console.error('Upload failed:', error);
    return res.status(500).json({ ok: false, error: 'Internal Server Error' });
  }
});

app.get('/health', (_req: Request, res: Response) => {
  res.json({ status: 'ok' });
});

app.listen(PORT, () => {
  // eslint-disable-next-line no-console
  console.log(`calimero-proxy server listening on http://localhost:${PORT}`);
  // eslint-disable-next-line no-console
  console.log(
    `POST /upload-model to upload ../trained-model.keras to Calimero context method upload_current_model`,
  );
  // eslint-disable-next-line no-console
  console.log(`Calimero RPC: ${API_BASE_URL}${JSONRPC_PATH}`);
  // eslint-disable-next-line no-console
  console.log(`Application/Context ID set: ${APPLICATION_ID ? 'yes' : 'no'}`);
});

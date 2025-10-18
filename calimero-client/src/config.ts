import { config } from 'dotenv';
import { CalimeroConfig } from './types.js';

// Load environment variables
config();

export function getConfig(): CalimeroConfig {
  // Support both VITE_ prefixed and non-prefixed environment variables
  const nodeUrl = process.env.VITE_NODE_URL || process.env.NODE_URL;
  const applicationId = process.env.VITE_APPLICATION_ID || process.env.APPLICATION_ID;
  const contextId = process.env.VITE_CONTEXT_ID || process.env.CONTEXT_ID;
  const executorPublicKey = process.env.VITE_EXECUTOR_PUBLIC_KEY || process.env.EXECUTOR_PUBLIC_KEY;
  const rpcPath = process.env.VITE_RPC_PATH || process.env.RPC_PATH || '/jsonrpc';
  const nearEnvironment = process.env.VITE_NEAR_ENVIRONMENT || process.env.NEAR_ENVIRONMENT || 'testnet';
  const cnnApiUrl = process.env.VITE_CNN_API_URL || process.env.CNN_API_URL;

  if (!nodeUrl) {
    throw new Error('VITE_NODE_URL is required in .env file');
  }

  if (!applicationId) {
    throw new Error('VITE_APPLICATION_ID is required in .env file');
  }

  if (!contextId) {
    throw new Error('VITE_CONTEXT_ID is required in .env file');
  }

  return {
    nodeUrl,
    applicationId,
    contextId,
    executorPublicKey,
    rpcPath,
    nearEnvironment,
    cnnApiUrl,
    accessToken: process.env.ACCESS_TOKEN,
    refreshToken: process.env.REFRESH_TOKEN,
  };
}


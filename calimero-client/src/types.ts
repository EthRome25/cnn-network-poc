export interface Upload {
  id: string;
  filename: string;
  url: string;
  size: number;
  mimeType?: string;
  uploadedAt: number;
  uploadedBy?: string;
}

export interface CreateUploadInput {
  filename: string;
  url: string;
  size: number;
  mimeType?: string;
}

export interface ListUploadsResponse {
  uploads: Upload[];
}

export interface CalimeroConfig {
  nodeUrl: string;
  applicationId: string;
  contextId: string;
  executorPublicKey?: string;
  rpcPath?: string;
  nearEnvironment?: string;
  cnnApiUrl?: string;
  accessToken?: string;
  refreshToken?: string;
}

export interface RpcResponse<T> {
  result?: T;
  error?: {
    code?: number;
    message: string;
  };
}


import axios, { AxiosInstance } from 'axios';
import {
  Upload,
  CreateUploadInput,
  ListUploadsResponse,
  CalimeroConfig,
} from './types.js';

interface RpcRequest {
  jsonrpc: string;
  id: number;
  method: string;
  params: {
    contextId: string;
    method: string;
    argsJson: any;
    executorPublicKey: string;
  };
}

interface RpcResponse<T> {
  jsonrpc: string;
  id: number;
  result?: {
    output?: T;
  };
  error?: {
    code: number;
    message: string;
    data?: any;
  };
}

export class SimpleCalimeroClient {
  private client: AxiosInstance;
  private config: CalimeroConfig;
  private requestId: number = 1;

  constructor(config: CalimeroConfig) {
    this.config = config;
    this.client = axios.create({
      baseURL: config.nodeUrl,
      headers: {
        'Content-Type': 'application/json',
      },
    });
  }

  // Make executeRpc public for direct access
  async executeRpc<TArgs = any, TResult = any>(method: string, args: TArgs): Promise<TResult> {
    const request: RpcRequest = {
      jsonrpc: '2.0',
      id: this.requestId++,
      method: 'execute',
      params: {
        contextId: this.config.contextId,
        method,
        argsJson: args,
        executorPublicKey: this.config.executorPublicKey || '',
      },
    };

    try {
      const response = await this.client.post<RpcResponse<TResult>>('/jsonrpc', request);

      if (response.data.error) {
        const errorMsg = response.data.error.message || JSON.stringify(response.data.error);
        console.error('RPC Error details:', response.data.error);
        throw new Error(`RPC Error: ${errorMsg}`);
      }

      if (!response.data.result || response.data.result.output === undefined) {
        console.error('Full response:', JSON.stringify(response.data, null, 2));
        throw new Error('No result returned from RPC');
      }

      return response.data.result.output;
    } catch (error) {
      if (axios.isAxiosError(error)) {
        if (error.response) {
          console.error('HTTP Error response:', error.response.data);
        }
        throw new Error(`HTTP Error: ${error.message}`);
      }
      throw error;
    }
  }

  async addUpload(input: CreateUploadInput): Promise<Upload> {
    console.log(`📤 Adding upload to Calimero context: ${input.filename}`);
    
    const upload = await this.executeRpc<CreateUploadInput, Upload>('add_upload', input);
    
    console.log('✅ Upload added successfully!');
    console.log(`   ID: ${upload.id}`);
    console.log(`   URL: ${upload.url}`);
    
    return upload;
  }

  async listUploads(): Promise<Upload[]> {
    console.log('📋 Fetching uploads from Calimero context...');
    
    const result = await this.executeRpc<{}, ListUploadsResponse>('list_uploads', {});
    
    console.log(`✅ Found ${result.uploads.length} uploads`);
    return result.uploads;
  }

  async getUpload(uploadId: string): Promise<Upload | null> {
    console.log(`🔍 Fetching upload: ${uploadId}`);
    
    try {
      const upload = await this.executeRpc<{ id: string }, Upload | null>('get_upload', { id: uploadId });
      
      if (upload) {
        console.log('✅ Upload found!');
        return upload;
      } else {
        console.log('⚠️  Upload not found');
        return null;
      }
    } catch (error) {
      console.log('⚠️  Upload not found');
      return null;
    }
  }

  async deleteUpload(uploadId: string): Promise<boolean> {
    console.log(`🗑️  Deleting upload: ${uploadId}`);
    
    const result = await this.executeRpc<{ id: string }, { success: boolean }>('delete_upload', { id: uploadId });
    
    if (result.success) {
      console.log('✅ Upload deleted successfully!');
    } else {
      console.log('⚠️  Upload not found or already deleted');
    }
    
    return result.success;
  }
}


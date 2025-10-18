import type { RpcClient } from '@calimero-network/calimero-client/lib/types/rpc.js';
import type { SubscriptionsClient } from '@calimero-network/calimero-client/lib/subscriptions/subscriptions.js';
import {
  Upload,
  CreateUploadInput,
  ListUploadsResponse,
  CalimeroConfig,
  RpcResponse,
} from './types.js';

export class CalimeroUploadClient {
  private rpcClient: RpcClient;
  private wsClient: SubscriptionsClient | null = null;
  private config: CalimeroConfig;

  constructor(config: CalimeroConfig, rpcClient: RpcClient) {
    this.config = config;
    this.rpcClient = rpcClient;
  }

  /**
   * Add a new upload to the Calimero context
   */
  async addUpload(input: CreateUploadInput): Promise<Upload> {
    try {
      console.log(`📤 Adding upload to Calimero context: ${input.filename}`);

      const response = await this.rpcClient.execute<CreateUploadInput, Upload>({
        contextId: this.config.contextId,
        method: 'add_upload',
        argsJson: input,
        executorPublicKey: this.config.executorPublicKey || '',
      });

      if (response.error) {
        throw new Error(`RPC Error: ${response.error.error.name}`);
      }

      if (!response.result || !response.result.output) {
        throw new Error('No result returned from add_upload');
      }

      const upload = response.result.output;
      console.log('✅ Upload added successfully!');
      console.log(`   ID: ${upload.id}`);
      console.log(`   URL: ${upload.url}`);

      return upload;
    } catch (error) {
      console.error('❌ Failed to add upload:', error);
      throw error;
    }
  }

  /**
   * List all uploads from the Calimero context
   */
  async listUploads(): Promise<Upload[]> {
    try {
      console.log('📋 Fetching uploads from Calimero context...');

      const response = await this.rpcClient.execute<{}, ListUploadsResponse>({
        contextId: this.config.contextId,
        method: 'list_uploads',
        argsJson: {},
        executorPublicKey: this.config.executorPublicKey || '',
      });

      if (response.error) {
        throw new Error(`RPC Error: ${response.error.error.name}`);
      }

      if (!response.result || !response.result.output) {
        throw new Error('No result returned from list_uploads');
      }

      console.log(`✅ Found ${response.result.output.uploads.length} uploads`);
      return response.result.output.uploads;
    } catch (error) {
      console.error('❌ Failed to list uploads:', error);
      throw error;
    }
  }

  /**
   * Get a specific upload by ID
   */
  async getUpload(uploadId: string): Promise<Upload | null> {
    try {
      console.log(`🔍 Fetching upload: ${uploadId}`);

      const response = await this.rpcClient.execute<{ id: string }, Upload | null>({
        contextId: this.config.contextId,
        method: 'get_upload',
        argsJson: { id: uploadId },
        executorPublicKey: this.config.executorPublicKey || '',
      });

      if (response.error) {
        throw new Error(`RPC Error: ${response.error.error.name}`);
      }

      if (response.result && response.result.output) {
        console.log('✅ Upload found!');
        return response.result.output;
      } else {
        console.log('⚠️  Upload not found');
        return null;
      }
    } catch (error) {
      console.error('❌ Failed to get upload:', error);
      throw error;
    }
  }

  /**
   * Delete an upload by ID
   */
  async deleteUpload(uploadId: string): Promise<boolean> {
    try {
      console.log(`🗑️  Deleting upload: ${uploadId}`);

      const response = await this.rpcClient.execute<{ id: string }, { success: boolean }>({
        contextId: this.config.contextId,
        method: 'delete_upload',
        argsJson: { id: uploadId },
        executorPublicKey: this.config.executorPublicKey || '',
      });

      if (response.error) {
        throw new Error(`RPC Error: ${response.error.error.name}`);
      }

      const success = response.result?.output?.success ?? false;
      
      if (success) {
        console.log('✅ Upload deleted successfully!');
      } else {
        console.log('⚠️  Upload not found or already deleted');
      }

      return success;
    } catch (error) {
      console.error('❌ Failed to delete upload:', error);
      throw error;
    }
  }

  /**
   * Subscribe to real-time updates from the context
   */
  async subscribeToUpdates(wsClient: SubscriptionsClient, callback: (event: any) => void): Promise<void> {
    try {
      console.log('🔌 Connecting to WebSocket...');

      this.wsClient = wsClient;
      await this.wsClient.connect();

      console.log('✅ WebSocket connected!');
      console.log(`📡 Subscribing to context: ${this.config.contextId}`);

      this.wsClient.subscribe([this.config.contextId]);

      this.wsClient.addCallback((event) => {
        console.log('\n📨 Received event:', event.type);
        
        if (event.type === 'StateMutation') {
          console.log('   State updated:', event.data.newRoot);
        } else if (event.type === 'ExecutionEvent') {
          console.log('   Execution events:', event.data.events);
        }

        callback(event);
      });

      console.log('🎧 Listening for updates... (Press Ctrl+C to stop)');
    } catch (error) {
      console.error('❌ Failed to subscribe to updates:', error);
      throw error;
    }
  }

  /**
   * Disconnect WebSocket subscription
   */
  disconnect(): void {
    if (this.wsClient) {
      console.log('👋 Disconnecting WebSocket...');
      this.wsClient.disconnect();
      this.wsClient = null;
    }
  }
}


#!/usr/bin/env node

import { program } from 'commander';
import { getConfig } from './config.js';
import { SimpleCalimeroClient } from './simpleClient.js';
import { CreateUploadInput } from './types.js';

async function main() {
  program
    .name('calimero-upload-client')
    .description('CLI tool to manage file uploads in Calimero context')
    .version('1.0.0');

  // Config command
  program
    .command('config')
    .description('Display current configuration')
    .action(() => {
      try {
        const config = getConfig();
        console.log('\n📋 Current Configuration:');
        console.log('─'.repeat(60));
        console.log(`Node URL:          ${config.nodeUrl}`);
        console.log(`RPC Path:          ${config.rpcPath || '/jsonrpc'}`);
        console.log(`Application ID:    ${config.applicationId}`);
        console.log(`Context ID:        ${config.contextId}`);
        console.log(`NEAR Environment:  ${config.nearEnvironment || 'testnet'}`);
        if (config.cnnApiUrl) {
          console.log(`CNN API URL:       ${config.cnnApiUrl}`);
        }
        if (config.executorPublicKey) {
          console.log(`Executor Key:      ${config.executorPublicKey.substring(0, 20)}...`);
        }
        console.log('─'.repeat(60));
      } catch (error) {
        console.error('Error:', error instanceof Error ? error.message : error);
        process.exit(1);
      }
    });

  // Test connection command
  program
    .command('test')
    .description('Test connection to Calimero context')
    .action(async () => {
      try {
        const config = getConfig();
        const client = new SimpleCalimeroClient(config);

        console.log('\n🔍 Testing connection to Calimero context...\n');
        console.log(`Node URL: ${config.nodeUrl}`);
        console.log(`Context ID: ${config.contextId}\n`);

        // Try to call get_current_model to test connection
        console.log('📞 Calling get_current_model...');
        
        const model = await client.executeRpc('get_current_model', {});
        
        if (model) {
          console.log('\n✅ Connection successful!');
          console.log('\n📄 Current Model in Context:');
          console.log('─'.repeat(60));
          console.log(`ID: ${model.id}`);
          console.log(`Name: ${model.name}`);
          console.log(`Version: ${model.version}`);
          console.log(`Type: ${model.model_type}`);
          console.log(`Size: ${model.file_size} bytes`);
          console.log(`Accuracy: ${model.prediction_accuracy}`);
          console.log(`Created: ${new Date(model.created_at / 1000).toLocaleString()}`);
          console.log('─'.repeat(60));
        } else {
          console.log('\n✅ Connection successful!');
          console.log('ℹ️  No model is currently stored in this context.');
        }
      } catch (error) {
        console.error('\n❌ Connection failed!');
        console.error('Error:', error instanceof Error ? error.message : error);
        process.exit(1);
      }
    });

  // Model command - show current ML model
  program
    .command('model')
    .description('Display the current ML model in the context')
    .option('-j, --json', 'Output as JSON')
    .action(async (options) => {
      try {
        const config = getConfig();
        const client = new SimpleCalimeroClient(config);

        const model = await client.executeRpc('get_current_model', {});
        
        if (options.json) {
          console.log(JSON.stringify(model, null, 2));
          return;
        }

        if (model) {
          console.log('\n🧠 Current ML Model:');
          console.log('═'.repeat(70));
          console.log(`\n📊 ${model.name}`);
          console.log('─'.repeat(70));
          console.log(`ID:              ${model.id}`);
          console.log(`Version:         ${model.version}`);
          console.log(`Type:            ${model.model_type}`);
          console.log(`Description:     ${model.description}`);
          console.log(`\nSize:            ${(model.file_size / 1024).toFixed(2)} KB (${model.file_size} bytes)`);
          console.log(`Accuracy:        ${model.prediction_accuracy}%`);
          console.log(`Public:          ${model.is_public ? 'Yes' : 'No'}`);
          console.log(`Uploader:        ${model.uploader}`);
          console.log(`Created:         ${new Date(model.created_at / 1000000).toLocaleString()}`);
          
          if (model.model_params && model.model_params !== '{}') {
            console.log(`\nParameters:      ${model.model_params}`);
          }
          
          console.log('═'.repeat(70));
        } else {
          console.log('\nℹ️  No model is currently stored in this context.');
        }
      } catch (error) {
        console.error('Error:', error instanceof Error ? error.message : error);
        process.exit(1);
      }
    });

  // Add upload command
  program
    .command('add')
    .description('Add a new upload to the context')
    .requiredOption('-f, --filename <filename>', 'Filename')
    .requiredOption('-u, --url <url>', 'Public URL of the file')
    .requiredOption('-s, --size <size>', 'File size in bytes', parseInt)
    .option('-m, --mime <mimeType>', 'MIME type')
    .action(async (options) => {
      try {
        const config = getConfig();
        const client = new SimpleCalimeroClient(config);

        const input: CreateUploadInput = {
          filename: options.filename,
          url: options.url,
          size: options.size,
          mimeType: options.mime,
        };

        const upload = await client.addUpload(input);
        console.log('\n📄 Upload details:');
        console.log(JSON.stringify(upload, null, 2));
      } catch (error) {
        console.error('Error:', error instanceof Error ? error.message : error);
        process.exit(1);
      }
    });

  // List uploads command
  program
    .command('list')
    .description('List all uploads in the context')
    .option('-j, --json', 'Output as JSON')
    .action(async (options) => {
      try {
        const config = getConfig();
        const client = new SimpleCalimeroClient(config);

        const uploads = await client.listUploads();

        if (options.json) {
          console.log(JSON.stringify(uploads, null, 2));
        } else {
          console.log('\n📂 Uploads:');
          console.log('─'.repeat(80));
          
          if (uploads.length === 0) {
            console.log('   (No uploads found)');
          } else {
            uploads.forEach((upload, index) => {
              console.log(`\n${index + 1}. ${upload.filename}`);
              console.log(`   ID: ${upload.id}`);
              console.log(`   URL: ${upload.url}`);
              console.log(`   Size: ${upload.size} bytes`);
              if (upload.mimeType) {
                console.log(`   Type: ${upload.mimeType}`);
              }
              console.log(`   Uploaded: ${new Date(upload.uploadedAt).toLocaleString()}`);
            });
          }
          
          console.log(`\n📊 Total: ${uploads.length} uploads`);
        }
      } catch (error) {
        console.error('Error:', error instanceof Error ? error.message : error);
        process.exit(1);
      }
    });

  // Get upload command
  program
    .command('get <id>')
    .description('Get a specific upload by ID')
    .action(async (id) => {
      try {
        const config = getConfig();
        const client = new SimpleCalimeroClient(config);

        const upload = await client.getUpload(id);

        if (upload) {
          console.log('\n📄 Upload details:');
          console.log(JSON.stringify(upload, null, 2));
        } else {
          console.log('❌ Upload not found');
          process.exit(1);
        }
      } catch (error) {
        console.error('Error:', error instanceof Error ? error.message : error);
        process.exit(1);
      }
    });

  // Delete upload command
  program
    .command('delete <id>')
    .description('Delete an upload by ID')
    .action(async (id) => {
      try {
        const config = getConfig();
        const client = new SimpleCalimeroClient(config);

        const success = await client.deleteUpload(id);

        if (!success) {
          process.exit(1);
        }
      } catch (error) {
        console.error('Error:', error instanceof Error ? error.message : error);
        process.exit(1);
      }
    });

  // Subscribe to updates command
  program
    .command('subscribe')
    .description('Subscribe to real-time updates from the context (WebSocket support coming soon)')
    .action(async () => {
      console.log('⚠️  WebSocket subscription support is not yet available in the simple client.');
      console.log('This feature will be added in a future update.');
      process.exit(0);
    });

  // Parse command line arguments
  program.parse();
}

// Handle uncaught errors
process.on('unhandledRejection', (error) => {
  console.error('Unhandled error:', error);
  process.exit(1);
});

// Run main if this is the entry point
main().catch((error) => {
  console.error('Fatal error:', error);
  process.exit(1);
});


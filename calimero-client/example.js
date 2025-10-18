/**
 * Example script showing how to use the Calimero Upload Client
 * 
 * This demonstrates the complete workflow:
 * 1. Initialize the client
 * 2. Add an upload
 * 3. List all uploads
 * 4. Get a specific upload
 * 5. Subscribe to real-time updates
 */

import { CalimeroUploadClient } from './dist/client.js';
import { config } from 'dotenv';

// Load environment variables
config();

async function main() {
  try {
    console.log('🚀 Calimero Upload Client Example\n');

    // Initialize the client
    const client = new CalimeroUploadClient({
      nodeUrl: process.env.NODE_URL,
      applicationId: process.env.APPLICATION_ID,
      contextId: process.env.CONTEXT_ID,
    });

    console.log('📋 Step 1: Adding a new upload...\n');
    
    // Add a new upload
    const newUpload = await client.addUpload({
      filename: 'example-document.pdf',
      url: 'http://localhost:9000/shared-files/example-document.pdf',
      size: 2048576, // 2MB
      mimeType: 'application/pdf',
    });

    console.log('\n✅ Upload added:', newUpload);

    console.log('\n📋 Step 2: Listing all uploads...\n');
    
    // List all uploads
    const uploads = await client.listUploads();
    
    console.log(`\nFound ${uploads.length} uploads:`);
    uploads.forEach((upload, index) => {
      console.log(`  ${index + 1}. ${upload.filename} (${upload.size} bytes)`);
    });

    console.log('\n📋 Step 3: Getting specific upload...\n');
    
    // Get the upload we just created
    const retrieved = await client.getUpload(newUpload.id);
    
    if (retrieved) {
      console.log('Retrieved upload:', retrieved);
    }

    console.log('\n📋 Step 4: Subscribing to real-time updates...\n');
    
    // Subscribe to updates (this will keep running)
    await client.subscribeToUpdates((event) => {
      console.log('📨 Received event:', event);
    });

    // Keep the process running
    console.log('\n🎧 Listening for updates... (Press Ctrl+C to stop)');

  } catch (error) {
    console.error('❌ Error:', error);
    process.exit(1);
  }
}

// Handle graceful shutdown
process.on('SIGINT', () => {
  console.log('\n\n👋 Shutting down...');
  process.exit(0);
});

main();


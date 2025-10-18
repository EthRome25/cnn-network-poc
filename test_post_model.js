#!/usr/bin/env node

import { getConfig } from './calimero-client/dist/config.js';
import { SimpleCalimeroClient } from './calimero-client/dist/simpleClient.js';

async function testPostModel() {
  try {
    console.log('🧪 Testing POST /model endpoint...');
    
    // Get configuration
    const config = getConfig();
    const client = new SimpleCalimeroClient(config);
    
    // Create test model data
    const testModelData = "Test_Model_v2.0.0_Data";
    const modelBase64 = Buffer.from(testModelData).toString('base64');
    
    console.log(`✅ Test model data prepared (${testModelData.length} bytes)`);
    
    // Prepare test model data
    const modelPayload = {
      name: "Test_Model_v2",
      description: "Test model to verify POST /model endpoint functionality",
      model_type: "test_classifier",
      version: "2.0.0",
      file_bytes_base64: modelBase64,
      uploader: "test_user",
      prediction_accuracy: 0.95,
      date: Date.now(),
      model_params: JSON.stringify({
        test: true,
        endpoint: "POST /model",
        verification: "successful"
      }),
      is_public: true
    };
    
    console.log('📤 Testing model upload via POST /model...');
    console.log(`   Name: ${modelPayload.name}`);
    console.log(`   Type: ${modelPayload.model_type}`);
    console.log(`   Version: ${modelPayload.version}`);
    console.log(`   Size: ${testModelData.length} bytes`);
    
    // Test the upload
    const result = await client.executeRpc('upload_current_model', modelPayload);
    
    console.log('✅ POST /model test successful!');
    console.log(`   Model ID: ${result}`);
    
    // Verify the upload
    console.log('\n🔍 Verifying upload...');
    const currentModel = await client.executeRpc('get_current_model', {});
    
    if (currentModel && currentModel.name === 'Test_Model_v2') {
      console.log('✅ Model verification successful!');
      console.log(`   Current model: ${currentModel.name}`);
      console.log(`   Version: ${currentModel.version}`);
      console.log(`   Type: ${currentModel.model_type}`);
      console.log(`   Size: ${currentModel.file_size} bytes`);
      console.log('\n🎉 POST /model endpoint is working correctly!');
    } else {
      console.log('❌ Model verification failed');
      console.log('Current model:', currentModel);
    }
    
  } catch (error) {
    console.error('❌ POST /model test failed:', error.message);
    console.error('Full error:', error);
    process.exit(1);
  }
}

// Run the test
testPostModel();

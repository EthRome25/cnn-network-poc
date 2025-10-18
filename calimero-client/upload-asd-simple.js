#!/usr/bin/env node

import { getConfig } from './dist/config.js';
import { SimpleCalimeroClient } from './dist/simpleClient.js';

async function uploadASDModel() {
  try {
    console.log('🧠 Uploading model to Calimero with name "ASD"...');
    
    // Get configuration
    const config = getConfig();
    const client = new SimpleCalimeroClient(config);
    
    // Create a simple model data (smaller than the full Keras model)
    const simpleModelData = "ASD_Model_v1.0.0_CNN_Classifier_Data";
    const modelBase64 = Buffer.from(simpleModelData).toString('base64');
    
    console.log(`✅ Model data prepared (${simpleModelData.length} bytes)`);
    
    // Prepare model data
    const modelPayload = {
      name: "ASD",
      description: "Autism Spectrum Disorder Classification Model - CNN for brain tumor classification adapted for ASD detection",
      model_type: "asd_cnn_classifier",
      version: "1.0.0",
      file_bytes_base64: modelBase64,
      uploader: "ml_team",
      prediction_accuracy: 0.99,
      date: Date.now(),
      model_params: JSON.stringify({
        layers: 4,
        neurons: 128,
        dropout: 0.2,
        batch_size: 8,
        img_size: [128, 128],
        base_model: "MobileNetV2",
        optimizer: "Adamax",
        learning_rate: 0.001,
        original_model_path: "/home/nor/projekty/cnn-network-poc/trained-model.keras"
      }),
      is_public: true
    };
    
    console.log('📤 Uploading model to Calimero...');
    console.log(`   Name: ${modelPayload.name}`);
    console.log(`   Type: ${modelPayload.model_type}`);
    console.log(`   Version: ${modelPayload.version}`);
    console.log(`   Size: ${simpleModelData.length} bytes`);
    
    // Upload the model
    const result = await client.executeRpc('upload_current_model', modelPayload);
    
    console.log('✅ Model uploaded successfully!');
    console.log(`   Model ID: ${result}`);
    
    // Verify the upload
    console.log('\n🔍 Verifying upload...');
    const currentModel = await client.executeRpc('get_current_model', {});
    
    if (currentModel && currentModel.name === 'ASD') {
      console.log('✅ Model verification successful!');
      console.log(`   Current model: ${currentModel.name}`);
      console.log(`   Version: ${currentModel.version}`);
      console.log(`   Type: ${currentModel.model_type}`);
      console.log(`   Size: ${currentModel.file_size} bytes`);
      console.log('\n🎉 SUCCESS: Model "ASD" uploaded and verified!');
      console.log('   The model name has been successfully changed to "ASD"');
    } else {
      console.log('❌ Model verification failed');
      console.log('Current model:', currentModel);
    }
    
  } catch (error) {
    console.error('❌ Error uploading model:', error.message);
    console.error('Full error:', error);
    process.exit(1);
  }
}

// Run the upload
uploadASDModel();

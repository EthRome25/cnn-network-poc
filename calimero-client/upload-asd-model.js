#!/usr/bin/env node

import { getConfig } from './dist/config.js';
import { SimpleCalimeroClient } from './dist/simpleClient.js';
import fs from 'fs';
import path from 'path';

function bumpPatchVersion(v) {
  try {
    const parts = (v || '1.0.0').split('.').map(n => parseInt(n, 10));
    const major = isNaN(parts[0]) ? 1 : parts[0];
    const minor = isNaN(parts[1]) ? 0 : parts[1];
    const patch = isNaN(parts[2]) ? 0 : parts[2];
    return `${major}.${minor}.${patch + 1}`;
  } catch {
    return '1.0.1';
  }
}

async function uploadASDModel() {
  try {
    console.log('🧠 Uploading model to Calimero with name "ASD"...');

    // Get configuration
    const config = getConfig();
    const client = new SimpleCalimeroClient(config);

    // Model file path
    const modelPath = path.join(process.cwd(), '..', 'trained-model.keras');
    console.log(`📁 Model path: ${modelPath}`);

    // Check if model file exists
    if (!fs.existsSync(modelPath)) {
      console.error(`❌ Model file not found: ${modelPath}`);
      process.exit(1);
    }

    // Read and encode model file
    console.log('📖 Reading model file...');
    const modelData = fs.readFileSync(modelPath);
    const modelBase64 = modelData.toString('base64');
    console.log(`✅ Model file read successfully (${modelData.length} bytes)`);

    // Determine next version by bumping the patch of current model (or default)
    console.log('\n🔍 Checking current model version to auto-bump patch...');
    let nextVersion;
    try {
      const currentModel = await client.executeRpc('get_current_model', {});
      const baseVersion = (currentModel && currentModel.version) || '1.0.0';
      nextVersion = bumpPatchVersion(baseVersion);
      console.log(`   Current version: ${baseVersion} -> Next: ${nextVersion}`);
    } catch {
      nextVersion = bumpPatchVersion('1.0.0');
      console.log(`   No current model found. Using default next version: ${nextVersion}`);
    }

    // Prepare model data
    const modelPayload = {
      name: "ASD",
      description: "Autism Spectrum Disorder Classification Model - CNN for brain tumor classification adapted for ASD detection",
      model_type: "asd_cnn_classifier",
      version: nextVersion,
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
        learning_rate: 0.001
      }),
      is_public: true
    };

    console.log('📤 Uploading model to Calimero...');
    console.log(`   Name: ${modelPayload.name}`);
    console.log(`   Type: ${modelPayload.model_type}`);
    console.log(`   Version: ${modelPayload.version}`);
    console.log(`   Size: ${modelData.length} bytes`);

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

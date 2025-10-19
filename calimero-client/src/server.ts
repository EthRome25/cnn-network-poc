import express from 'express';
import cors from 'cors';
import {getConfig} from './config.js';
import {SimpleCalimeroClient} from './simpleClient.js';

const app = express();
const PORT = process.env.PORT || 3420;

// Middleware
app.use(cors());
app.use(express.json({limit: '50mb'})); // Increase payload limit for large model files

// Initialize Calimero client
const config = getConfig();
const client = new SimpleCalimeroClient(config);

function bumpPatchVersion(v: string): string {
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

/**
 * GET /model
 * Get the current ML model from the Calimero context
 */
app.get('/model', async (req, res) => {
    try {
        console.log('📥 GET /model - Fetching current model...');

        const model = await client.executeRpc('get_current_model', {});

        if (model) {
            console.log('✅ Model found:', model.name);
            res.json({
                success: true,
                data: model
            });
        } else {
            console.log('⚠️  No model found');
            res.json({
                success: true,
                data: null,
                message: 'No model is currently stored in the context'
            });
        }
    } catch (error) {
        console.error('❌ Error fetching model:', error);
        res.status(500).json({
            success: false,
            error: error instanceof Error ? error.message : 'Unknown error'
        });
    }
});

/**
 * POST /model
 * Upload a new ML model to the Calimero context
 *
 * Body:
 * {
 *   "name": "Model Name",
 *   "description": "Model description",
 *   "model_type": "classifier",
 *   "version": "1.0.0",
 *   "file_bytes_base64": "base64_encoded_data",
 *   "uploader": "user_name",
 *   "prediction_accuracy": 0.95,
 *   "date": 1697654400,
 *   "model_params": "{\"layers\": 3}",
 *   "is_public": true
 * }
 */
app.post('/model', async (req, res) => {
    try {
        console.log('📤 POST /model - Uploading new model...');

        const {
            name,
            description,
            model_type,
            version,
            file_bytes_base64,
            uploader,
            prediction_accuracy,
            date,
            model_params,
            is_public
        } = req.body;

        // Validate required fields (version is optional; server auto-bumps based on current model)
        if (!name || !description || !model_type || !file_bytes_base64 || !uploader) {
            return res.status(400).json({
                success: false,
                error: 'Missing required fields: name, description, model_type, file_bytes_base64, uploader'
            });
        }

        console.log(`   Model: ${name} v${version}`);
        console.log(`   Type: ${model_type}`);
        console.log(`   Size: ${file_bytes_base64.length} bytes`);
        console.log("Prediction accuracy: ", prediction_accuracy*100);

        // Determine the effective version by bumping the patch
        let effectiveVersion: string;
        try {
            const current = await client.executeRpc('get_current_model', {});
            const baseVersion = (current && current.version) || version || '1.0.0';
            effectiveVersion = bumpPatchVersion(baseVersion);
        } catch (e) {
            const baseVersion = version || '1.0.0';
            effectiveVersion = bumpPatchVersion(baseVersion);
        }

        console.log(`   Using version: ${effectiveVersion} (auto-bumped)`);

        const result = await client.executeRpc('upload_current_model', {
            name,
            description,
            model_type,
            version: effectiveVersion,
            file_bytes_base64,
            uploader,
            prediction_accuracy: prediction_accuracy*100,
            date: date ?? Date.now(),
            model_params: model_params || '{}',
            is_public: is_public !== undefined ? is_public : true
        });

        console.log('✅ Model uploaded successfully:', result);

        res.json({
            success: true,
            data: {
                model_id: result,
                message: 'Model uploaded successfully'
            }
        });
    } catch (error) {
        console.error('❌ Error uploading model:', error);
        res.status(500).json({
            success: false,
            error: error instanceof Error ? error.message : 'Unknown error'
        });
    }
});

/**
 * GET /health
 * Health check endpoint
 */
app.get('/health', (req, res) => {
    res.json({
        success: true,
        status: 'healthy',
        context: {
            contextId: config.contextId,
            nodeUrl: config.nodeUrl
        }
    });
});

/**
 * GET /
 * API info endpoint
 */
app.get('/', (req, res) => {
    res.json({
        name: 'Calimero ML Model API',
        version: '1.0.0',
        endpoints: {
            'GET /': 'API information',
            'GET /health': 'Health check',
            'GET /model': 'Get current ML model',
            'POST /model': 'Upload new ML model'
        },
        context: {
            contextId: config.contextId,
            applicationId: config.applicationId
        }
    });
});

// Start server
app.listen(PORT, () => {
    console.log('\n🚀 Calimero ML Model API Server');
    console.log('═'.repeat(50));
    console.log(`📡 Server running on: http://localhost:${PORT}`);
    console.log(`🌐 Context ID: ${config.contextId}`);
    console.log(`🔗 Node URL: ${config.nodeUrl}`);
    console.log('═'.repeat(50));
    console.log('\n📋 Available endpoints:');
    console.log(`   GET  http://localhost:${PORT}/`);
    console.log(`   GET  http://localhost:${PORT}/health`);
    console.log(`   GET  http://localhost:${PORT}/model`);
    console.log(`   POST http://localhost:${PORT}/model`);
    console.log('\n✅ Server is ready!\n');
});

// Handle graceful shutdown
process.on('SIGINT', () => {
    console.log('\n\n👋 Shutting down server...');
    process.exit(0);
});

process.on('SIGTERM', () => {
    console.log('\n\n👋 Shutting down server...');
    process.exit(0);
});


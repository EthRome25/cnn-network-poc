# Calimero Proxy

A minimal Express.js service that exposes a single POST endpoint to upload a local Keras model file (../trained-model.keras) to a Calimero context method named upload_current_model.

- Endpoint: POST /upload-model
- Health: GET /health

The server base64-encodes the model file and calls the Calimero JSON-RPC execute method with params:
- applicationId: your context/application ID
- method: "upload_current_model"
- argsJson: payload containing model metadata and the file content

## Prerequisites
- Node.js 18+ (for global fetch)
- The model file must exist at: ../trained-model.keras (relative to this project directory)

## Install and Run
- Install dependencies: pnpm install (or npm install / yarn)
- Set environment variables (examples below)
- Start the server: pnpm start (or npm start / yarn start)
- Default port: 2137

Health check:
- curl http://localhost:2137/health

## Environment Variables
The service reads multiple variable names for convenience. First non-empty value is used.

- API base URL (Calimero node):
  - CALIMERO_API_URL → NEXT_PUBLIC_API_URL → VITE_NODE_URL → default http://localhost:2428
- JSON-RPC path:
  - CALIMERO_JSONRPC_PATH → VITE_RPC_PATH → default /jsonrpc
- Application/Context ID:
  - CALIMERO_APPLICATION_ID → NEXT_PUBLIC_APPLICATION_ID → VITE_APPLICATION_ID → VITE_CONTEXT_ID
- Optional uploader label:
  - MODEL_UPLOADER (defaults to "server")

Example .env values (if you want to use VITE_*):
- VITE_APPLICATION_ID="HELDXwknx9tVnj3JKfa3EMyGB9JEsApeijVHzKn5cRVX"
- VITE_NEAR_ENVIRONMENT=testnet
- VITE_NODE_URL=http://localhost:2428
- VITE_RPC_PATH=/jsonrpc
- VITE_CONTEXT_ID=4bZJB5vmPAPn7yYwPDwDfJmpEbTKdFm3pvcmSpZMDuYx
- VITE_CNN_API_URL=http://localhost:8000

## Example Request Payload
The /upload-model endpoint accepts a JSON body. All fields are optional; sensible defaults are applied if omitted. The model file is always read from ../trained-model.keras.

Fields:
- name: string (default: "trained-model")
- description: string (default: "Uploaded via calimero-proxy")
- model_type: string (default: "keras")
- version: string (default: "1.0.0")
- uploader: string (default: value of MODEL_UPLOADER or "server")
- prediction_accuracy: number (float, default: 0.0)
- date: number (Unix epoch ms, default: Date.now())
- model_params: string (default: "")
- is_public: boolean (default: true)

These are mapped to the contract ABI fields for upload_current_model, with the model bytes injected as file_bytes_base64.

### JSON Example
{
  "name": "cnn-mnist",
  "description": "Convolutional NN trained on MNIST",
  "model_type": "keras",
  "version": "1.2.3",
  "uploader": "training-service",
  "prediction_accuracy": 0.9842,
  "date": 1734567890123,
  "model_params": "epochs=10,batch_size=32,optimizer=adam",
  "is_public": true
}

### curl Example
curl -X POST "http://localhost:2137/upload-model" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "cnn-mnist",
    "description": "Convolutional NN trained on MNIST",
    "model_type": "keras",
    "version": "1.2.3",
    "uploader": "training-service",
    "prediction_accuracy": 0.9842,
    "date": 1734567890123,
    "model_params": "epochs=10,batch_size=32,optimizer=adam",
    "is_public": true
  }'

Note: The server reads ../trained-model.keras and base64-encodes it as file_bytes_base64 internally; you do not send the file content in the request body.

## Responses
- 200 OK
  {
    "ok": true,
    "result": "<rpc_result_or_null>"
  }

- 400 Bad Request (validation / invalid params)
  {
    "ok": false,
    "error": { "code": -32602, "message": "..." }
  }

- 404 Model file not found
  {
    "ok": false,
    "error": "Model file not found at /path/to/../trained-model.keras"
  }

- 502 Upstream RPC error
  {
    "ok": false,
    "error": { "code": 502, "message": "..." }
  }

- 500 Internal Server Error
  {
    "ok": false,
    "error": "Internal Server Error"
  }

## Implementation Notes
- The server uses the official Calimero SDK JsonRpcClient to call method "execute" with params `{ applicationId, method, argsJson }` against `${API_BASE_URL}${JSONRPC_PATH}`. The path can be overridden via env (default `/jsonrpc`).
- Arguments are aligned with res/abi.json for the method upload_current_model.

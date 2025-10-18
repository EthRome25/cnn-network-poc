# Calimero Upload Client

A lightweight TypeScript client for managing file uploads in a Calimero decentralized context. This application connects to a Calimero node and provides a CLI interface to add, list, retrieve, and delete file uploads stored in a distributed context.

This client uses a simple, direct HTTP approach to communicate with the Calimero JSON-RPC API, avoiding complex dependency chains.

## Features

- ✅ Add file uploads to Calimero context
- ✅ List all uploads from the context
- ✅ Get specific upload by ID
- ✅ Delete uploads from context
- ✅ Real-time updates via WebSocket subscription
- ✅ Full TypeScript support with type definitions
- ✅ Built on Calimero SDK

## Prerequisites

- Node.js 18+ or higher
- Access to a running Calimero node
- Calimero application ID and context ID

## Installation

1. Install dependencies using pnpm:

```bash
pnpm install
```

2. Copy the example environment file:

```bash
cp .env.example .env
```

3. Configure your `.env` file with the provided values:

```env
VITE_APPLICATION_ID=HELDXwknx9tVnj3JKfa3EMyGB9JEsApeijVHzKn5cRVX
VITE_NEAR_ENVIRONMENT=testnet
VITE_NODE_URL=http://localhost:2428
VITE_RPC_PATH=/jsonrpc
VITE_CONTEXT_ID=4bZJB5vmPAPn7yYwPDwDfJmpEbTKdFm3pvcmSpZMDuYx
VITE_CNN_API_URL=http://localhost:8045
```

**Note:** The `.env` file is already configured with these values. Make sure your Calimero node is running on `http://localhost:2428`.

## Usage

### Development Mode (with TypeScript)

```bash
pnpm dev
```

### Build for Production

```bash
pnpm build
pnpm start
```

## CLI Commands

### Display Configuration

View your current configuration:

```bash
pnpm dev config
```

This will display:
- Node URL
- RPC Path
- Application ID
- Context ID
- NEAR Environment
- CNN API URL (if configured)
- Executor Public Key (if configured)

### Add an Upload

Add a new upload to the Calimero context:

```bash
pnpm dev add \
  --filename "example.pdf" \
  --url "http://localhost:9000/shared-files/example.pdf" \
  --size 1024000 \
  --mime "application/pdf"
```

Or using the built version:

```bash
pnpm start add \
  --filename "example.pdf" \
  --url "http://localhost:9000/shared-files/example.pdf" \
  --size 1024000 \
  --mime "application/pdf"
```

### List All Uploads

Display all uploads in the context:

```bash
pnpm dev list
```

For JSON output:

```bash
pnpm dev list --json
```

### Get a Specific Upload

Retrieve details of a specific upload by ID:

```bash
pnpm dev get <upload-id>
```

### Delete an Upload

Remove an upload from the context:

```bash
pnpm dev delete <upload-id>
```

### Subscribe to Real-time Updates

**Note:** WebSocket subscription support is planned for a future update. Currently, this feature is not available in the simplified client implementation.

## Integration with MinIO

This client is designed to work seamlessly with MinIO file storage. Here's an example workflow:

1. **Upload file to MinIO** (from your minio-file-host project):
   ```bash
   cd ../minio-file-host
   node upload.js upload my-file.pdf
   ```
   
2. **Copy the public URL** from MinIO output

3. **Add to Calimero context**:
   ```bash
   cd ../calimero-client
   pnpm dev add \
     --filename "my-file.pdf" \
     --url "http://localhost:9000/shared-files/my-file.pdf" \
     --size 2048000 \
     --mime "application/pdf"
   ```

## Application Structure

```
calimero-client/
├── src/
│   ├── index.ts         # Main CLI entry point
│   ├── simpleClient.ts  # Simple Calimero RPC client
│   ├── client.ts        # Advanced client (with full SDK integration)
│   ├── config.ts        # Configuration management
│   └── types.ts         # TypeScript type definitions
├── dist/                # Compiled JavaScript output
├── .env                 # Environment configuration
├── package.json         # Dependencies and scripts
├── tsconfig.json        # TypeScript configuration
└── README.md            # This file
```

## API Methods

The application expects the following methods to be implemented in your Calimero application:

### `add_upload`

Adds a new upload to the context.

**Input:**
```typescript
{
  filename: string;
  url: string;
  size: number;
  mimeType?: string;
}
```

**Output:**
```typescript
{
  id: string;
  filename: string;
  url: string;
  size: number;
  mimeType?: string;
  uploadedAt: number;
  uploadedBy?: string;
}
```

### `list_uploads`

Returns all uploads from the context.

**Input:** `{}`

**Output:**
```typescript
{
  uploads: Upload[];
}
```

### `get_upload`

Gets a specific upload by ID.

**Input:**
```typescript
{
  id: string;
}
```

**Output:** `Upload | null`

### `delete_upload`

Deletes an upload by ID.

**Input:**
```typescript
{
  id: string;
}
```

**Output:**
```typescript
{
  success: boolean;
}
```

## WebSocket Events

When subscribed, the client will receive real-time events:

- **StateMutation**: Triggered when the context state changes
- **ExecutionEvent**: Triggered when methods are executed

## Error Handling

All commands include comprehensive error handling:

- Invalid configuration will exit with helpful error messages
- RPC errors are caught and displayed with details
- Network errors are handled gracefully
- All errors include stack traces in development mode

## Development

### Type Safety

The project is fully typed with TypeScript. All API requests and responses are type-checked at compile time.

### Adding New Commands

To add new commands, edit `src/index.ts` and add a new `.command()` definition. Implement the corresponding logic in `src/client.ts`.

### Extending the Client

The `CalimeroUploadClient` class can be extended with additional methods to support more application-specific functionality.

## Troubleshooting

### Connection Issues

If you can't connect to the Calimero node:

1. Verify the node is running: `curl http://localhost:2428/admin-api/health`
2. Check your `.env` configuration
3. Ensure firewall rules allow connections

### Authentication Errors

If you receive authentication errors:

1. Ensure you have valid access tokens
2. Check that your application ID and context ID are correct
3. Verify you have permissions for the context

### RPC Method Not Found

If methods are not found:

1. Ensure your Calimero application is properly installed
2. Verify the application implements the required methods
3. Check the application ID matches your deployment

## Resources

- [Calimero Documentation](https://docs.calimero.network/)
- [TypeScript Client SDK Guide](https://docs.calimero.network/developer-tools/SDK/client-sdk/client-ts-sdk)
- [Calimero GitHub](https://github.com/calimero-network)

## License

MIT


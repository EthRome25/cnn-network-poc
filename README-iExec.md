# CNN Brain Tumor Classification - iExec TEE Deployment

This repository contains a CNN-based brain tumor classification service that has been dockerized for deployment on iExec's Trusted Execution Environment (TEE) platform.

## Overview

The application classifies brain tumor MRI images into four categories:
- **Glioma**: A type of tumor that occurs in the brain and spinal cord
- **Meningioma**: A tumor that forms on membranes that cover the brain and spinal cord
- **No Tumor**: Normal brain tissue
- **Pituitary**: A tumor in the pituitary gland

## iExec TEE Features

- **Privacy-Preserving**: Your MRI images and model predictions are processed in a secure enclave
- **Confidential Computing**: The AI model and data remain encrypted during processing
- **Decentralized**: Runs on iExec's distributed network of trusted workers
- **Verifiable**: All computations are cryptographically verifiable

## Prerequisites

1. **iExec CLI**: Install the iExec command-line interface
   ```bash
   npm install -g @iexec/iapp
   ```

2. **Docker**: Ensure Docker is installed and running
   ```bash
   docker --version
   ```

3. **DockerHub Account**: You'll need a DockerHub account to push images

## Quick Start

### 1. Initialize iExec Project

```bash
# Initialize the iExec project (if not already done)
iapp init --skip-wallet
```

### 2. Configure Your Settings

Edit the following files with your information:

- `iapp.config.json`: Update author, email, repository URL, and DockerHub username
- `iexec.json`: Update author, email, and repository information
- `sconify.sh`: Update the `DOCKERHUB_USERNAME` variable

### 3. Choose Your Deployment Method

#### Option A: Full TEE Deployment (Recommended for Production)
```bash
# Set your DockerHub username
export DOCKERHUB_USERNAME=your-actual-username

# Run the complete deployment script
./deploy-to-iexec.sh
```

#### Option B: Alternative TEE Deployment (With Fallbacks)
```bash
# Set your DockerHub username
export DOCKERHUB_USERNAME=your-actual-username

# Run the alternative deployment script
./deploy-to-iexec-alternative.sh
```

#### Option C: Simple Deployment (No TEE - For Testing)
```bash
# Set your DockerHub username
export DOCKERHUB_USERNAME=your-actual-username

# Run the simple deployment script
./deploy-simple.sh
```

### 4. Test Your Deployment

```bash
# Test locally first
iapp test --args "input_image_url=https://example.com/sample-mri.jpg"

# Run on iExec network
iapp run --args "input_image_url=https://example.com/sample-mri.jpg"
```

## API Endpoints

The deployed service provides the following endpoints:

### Health Check
- **GET** `/` - Returns service status and usage information

### Prediction
- **POST** `/predict` - Classify an uploaded MRI image
  - **Input**: Multipart form data with image file
  - **Output**: JSON with prediction results and visualization

### Retraining (Optional)
- **POST** `/retrain` - Retrain the model with new data
  - **Input**: JSON with training parameters
  - **Output**: Training results and updated model

## Input/Output Format

### Input
- **Image Format**: JPG, PNG, or other common image formats
- **Size**: Images are automatically resized to 128x128 pixels
- **Channels**: RGB (3 channels)

### Output
```json
{
  "predicted_label": "meningioma",
  "probabilities": {
    "glioma": 0.02,
    "meningioma": 0.93,
    "notumor": 0.01,
    "pituitary": 0.04
  },
  "plot_base64_png": "iVBORw0KGgoAAA..."
}
```

## Configuration Files

### Dockerfile
- Optimized for iExec TEE environment
- Uses Python 3.11 slim base image
- Includes all necessary dependencies
- Runs as non-root user for security

### sconify.sh
- Creates TEE-compatible Docker image using SCONE framework
- Configures secure enclave parameters
- Generates cryptographically signed image

### iexec.json
- Defines app orders and pricing
- Configures TEE requirements
- Sets up request parameters

### chain.json
- Configures supported blockchain networks
- Default: Bellecour testnet
- Also supports Arbitrum mainnet

### iapp.config.json
- Application metadata and configuration
- Defines input/output specifications
- Resource requirements

## Security Features

- **TEE Protection**: Code and data are protected by Intel SGX enclaves
- **Encrypted Processing**: All computations happen in encrypted memory
- **Attestation**: Cryptographic proof of execution integrity
- **No Data Leakage**: Input data cannot be accessed by the worker

## Pricing and Economics

- **App Price**: 1 RLC per execution (configurable in `iexec.json`)
- **Volume**: 1000 authorized executions
- **TEE Tag**: Required for confidential computing

## Troubleshooting

### Common Issues

1. **Docker Build Fails**
   - Ensure all dependencies are in `requirements.txt`
   - Check that the trained model file exists
   - Verify Docker is running and accessible

2. **SCONE/TEE Build Fails**
   - **Registry Access Denied**: The SCONE registry may require authentication or have access restrictions
   - **Network Issues**: Check your internet connection and firewall settings
   - **Alternative**: Use the iExec CLI sconify command: `iapp sconify --from your-image:tag`
   - **Fallback**: Deploy without TEE using `./deploy-simple.sh`

3. **Deployment Fails**
   - Check iExec CLI configuration: `iapp config show`
   - Verify wallet has sufficient RLC tokens
   - Ensure TEE workerpools are available: `iapp workerpool list`
   - Check if you're using the correct chain: `iapp config show`

4. **Command Not Found: iapp**
   - Install the iExec CLI: `npm install -g @iexec/iapp`
   - Verify installation: `iapp --version`

5. **DockerHub Push Fails**
   - Login to DockerHub: `docker login`
   - Verify your DockerHub username is correct
   - Check if the repository exists or create it on DockerHub

### Debug Commands

```bash
# Check iExec configuration
iapp config show

# View available workerpools
iapp workerpool list

# Check app deployment status
iapp show

# View task execution logs
iapp task show <task-id>

# Check Docker images
docker images | grep brain-tumor-cnn

# Test Docker image locally
docker run --rm -p 8000:8000 your-username/brain-tumor-cnn:0.0.1
```

### TEE-Specific Troubleshooting

If you encounter issues with TEE deployment:

1. **Try the alternative deployment script**:
   ```bash
   ./deploy-to-iexec-alternative.sh
   ```

2. **Use simple deployment for testing**:
   ```bash
   ./deploy-simple.sh
   ```

3. **Manual TEE creation**:
   ```bash
   # After pushing your image to DockerHub
   iapp sconify --from your-username/brain-tumor-cnn:0.0.1
   iapp deploy
   ```

4. **Check TEE workerpool availability**:
   ```bash
   iapp workerpool list --tag 0x0000000000000000000000000000000000000000000000000000000000000003
   ```

## Development

### Local Testing

```bash
# Build and run locally
docker build -t brain-tumor-cnn .
docker run -p 8000:8000 brain-tumor-cnn

# Test with curl
curl -X POST -F "file=@sample-mri.jpg" http://localhost:8000/predict
```

### Model Updates

To update the model:
1. Replace `trained-model.keras` with your new model
2. Ensure the new model has the same input/output format
3. Rebuild and redeploy the Docker image

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
- Create an issue in this repository
- Check iExec documentation: https://docs.iex.ec
- Join the iExec community: https://t.me/iexec

## Acknowledgments

- iExec team for the TEE infrastructure
- TensorFlow team for the machine learning framework
- The medical imaging community for the dataset

# FastAPI Backend - Skin Lesion Classifier

## Local Development

### Prerequisites
- Python 3.8+
- PyTorch with CUDA support (optional)
- best_model.pth file in parent directory

### Installation

```bash
cd backend
pip install -r requirements.txt
```

### Running the Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### Health Check
```
GET /health
```

#### Prediction
```
POST /predict
Content-Type: multipart/form-data
Body: file (image)
```

## Deployment

### Railway (Recommended for GPU Support)

1. Create a Railway account at [railway.app](https://railway.app)
2. Install Railway CLI: `npm install -g @railway/cli`
3. Login: `railway login`
4. Initialize project: `railway init`
5. Deploy: `railway up`

For GPU support, you may need to contact Railway for access to GPU instances.

### Render

1. Create a Render account at [render.com](https://render.com)
2. Create a new Web Service
3. Connect your GitHub repository
4. Select the `backend` folder as root directory
5. Use the Dockerfile for build
6. Deploy

Note: Render does not currently offer GPU support for standard plans.

### Alternative: VPS with GPU

For full GPU support, consider:
- AWS EC2 with GPU instances
- Google Cloud Platform with GPU
- DigitalOcean GPU Droplets
- Linode GPU instances

## Environment Variables

- No environment variables required for basic operation
- CORS is currently set to allow all origins (configure for production)

## Model File

Ensure `best_model.pth` is in the parent directory or update the `MODEL_PATH` in `main.py`.

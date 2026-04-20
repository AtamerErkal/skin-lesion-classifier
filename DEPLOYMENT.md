# SkinXAI Deployment Guide

## 🚀 Hızlı Deploy Seçenekleri

### Seçenek 1: Vercel (Frontend) + Render/Railway (Backend) - ÜCRETSİZ

#### 1. Frontend (Next.js) - Vercel

```bash
# Vercel CLI ile deploy
cd frontend
npx vercel

# Veya GitHub repo bağla:
# 1. GitHub'a push et
# 2. vercel.com'dan "Add New Project"
# 3. Repo seç
# 4. Build ayarları otomatik algılanır
```

**Vercel Ortam Değişkenleri:**
```
NEXT_PUBLIC_API_URL=https://skinxai-api.onrender.com
```

#### 2. Backend (FastAPI) - Render

```bash
# Render'a deploy etmek için:
# 1. GitHub repo'sunu Render'a bağla
# 2. "New Web Service" > GitHub repo
# 3. Ayarlar:
#    - Build Command: pip install -r requirements.txt
#    - Start Command: uvicorn main:app --host 0.0.0.0 --port $PORT
#    - Root Directory: backend/
# 4. Environment: Python 3
```

**Render Ortam Değişkenleri:**
```
PORT=8001
```

---

### Seçenek 2: Railway (Her ikisi birlikte) - ÜCRETSİZ

```bash
# Railway CLI
npm i -g @railway/cli

# Login
railway login

# Backend deploy
cd backend
railway init
railway up

# Frontend deploy  
cd ../frontend
railway init
railway up
```

---

### Seçenek 3: Netlify (Frontend) + Hugging Face (Backend) - ÜCRETSİZ

#### Frontend - Netlify
```bash
cd frontend
npm run build
# Drag & drop `out/` klasörünü Netlify'e
```

#### Backend - Hugging Face Spaces
```python
# backend/app.py olarak Hugging Face formatında düzenle
# Space type: Gradio veya Docker
```

---

## 🔧 Adım Adım Deploy Talimatları

### Backend Deploy (FastAPI)

#### A) Render.com (Önerilen)

1. **GitHub Repo Push:**
```bash
git add .
git commit -m "SkinXAI v1.0"
git push origin main
```

2. **Render Dashboard:**
   - [dashboard.render.com](https://dashboard.render.com)
   - "New" → "Web Service"
   - GitHub repo bağla
   - **Build Command:** `pip install -r backend/requirements.txt`
   - **Start Command:** `cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Plan:** Free

3. **Wait for deploy** → URL al: `https://skinxai-api.onrender.com`

#### B) Railway.app

1. **Dashboard:** [railway.app](https://railway.app)
2. "New Project" → "Deploy from GitHub repo"
3. **Settings:**
   - Root Directory: `backend`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port 8001`
4. **Deploy!**

---

### Frontend Deploy (Next.js)

#### A) Vercel (En kolay)

```bash
# 1. Vercel CLI kur (eğer yoksa)
npm i -g vercel

# 2. Deploy
cd frontend
vercel --prod
```

**Otomatik GitHub Deploy:**
1. GitHub repo push
2. [vercel.com](https://vercel.com) → "Add New Project"
3. Repo seç
4. Framework: Next.js (otomatik algılar)
5. Build Command: `npm run build`
6. Output Directory: `.next`
7. **Environment Variable:**
   - `NEXT_PUBLIC_API_URL` = Backend URL

#### B) Netlify

```bash
cd frontend

# next.config.js düzenle:
# output: 'export', distDir: 'dist'

npm run build
# `dist/` klasörünü Netlify'e sürükle-bırak
```

---

## 📋 Önemli Notlar

### Backend için Model Dosyası

**Problem:** `best_model.pth` (~250MB) GitHub'a push edilemeyebilir.

**Çözümler:**

1. **Git LFS kullan:**
```bash
git lfs track "*.pth"
git add .gitattributes
```

2. **Veya Cloud Storage'dan indir:**
```python
# backend/main.py'de model loading'i değiştir:
import requests
import os

MODEL_URL = "https://your-storage.com/best_model.pth"
MODEL_PATH = "./best_model.pth"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)
```

### CORS Ayarları

Backend'de CORS domain'ini güncelle:

```python
# backend/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://skinxai.vercel.app",     # Vercel URL
        "https://skinxai.netlify.app",    # Netlify URL
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 🌐 Production URLs

| Bileşen | Ücretsiz Platform | URL Pattern |
|---------|------------------|-------------|
| Frontend | Vercel | `https://skinxai.vercel.app` |
| Frontend | Netlify | `https://skinxai.netlify.app` |
| Backend | Render | `https://skinxai-api.onrender.com` |
| Backend | Railway | `https://skinxai.up.railway.app` |

---

## ⚡ Hızlı Başlangıç (Copy-Paste)

```bash
# 1. GitHub'a push
git init
git add .
git commit -m "SkinXAI v1.0"
git remote add origin https://github.com/YOUR_USERNAME/SkinXAI.git
git push -u origin main

# 2. Backend Deploy - Render
# Render dashboard'a git, repo bağla, deploy et

# 3. Frontend Deploy - Vercel
cd frontend
npx vercel --prod

# 4. API URL'yi güncelle
# Vercel Dashboard > Project Settings > Environment Variables
# NEXT_PUBLIC_API_URL = https://skinxai-api.onrender.com
```

---

## 🔒 Güvenlik

Production'da şunları ekle:

```python
# backend/main.py
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/predict")
@limiter.limit("10/minute")  # Rate limiting
async def predict(request: Request, file: UploadFile = File(...)):
    ...
```

---

## 📱 Domain ve SSL

Vercel ve Render **otomatik SSL (HTTPS)** sağlar.

Özel domain eklemek için:
- Vercel: Settings > Domains
- Render: Settings > Custom Domains

---

**Hazır!** 🎉 SkinXAI deploy edildi: `https://skinxai.vercel.app`

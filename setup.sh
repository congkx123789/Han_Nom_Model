#!/bin/bash

# Hán-Nôm Heritage Platform Setup Script
echo "🏗️ Initializing Heritage Platform..."

# 1. Start Infrastructure
if [ -x "$(command -v docker-compose)" ]; then
    echo "🐳 Starting Docker infrastructure (PostgreSQL, MinIO, Redis, Milvus)..."
    docker-compose up -d
else
    echo "⚠️ docker-compose not found. Please start infrastructure manually."
fi

# 2. Setup Backend
echo "🐍 Setting up Python Backend..."
cd backend
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install -r requirements.txt
echo "✅ Backend ready. Run 'uvicorn app.main:app --reload' to start."
cd ..

# 3. Setup Frontend
echo "⚛️ Setting up React Frontend..."
cd frontend
npm install
echo "✅ Frontend ready. Run 'npm run dev' to start."
cd ..

echo "🚀 Setup Complete! See README.md for more details."

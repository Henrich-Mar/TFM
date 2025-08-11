#!/bin/bash

# Terraforming Mars RL Training Startup Script

echo "🚀 Starting Terraforming Mars RL Training Environment"
echo "=================================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Create necessary directories
mkdir -p rl-models
mkdir -p rl-logs

# Stop any existing containers
echo "🧹 Cleaning up existing containers..."
docker-compose -f docker-compose.rl.yml down

# Build and start the RL environment
echo "🏗️ Building and starting RL environment..."
docker-compose -f docker-compose.rl.yml up --build -d

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 30

# Check service health
echo "🔍 Checking service health..."

# Check game servers
for port in 8081 8082 8083; do
    if curl -f -s "http://localhost:$port/" > /dev/null; then
        echo "✅ Game server on port $port is running"
    else
        echo "❌ Game server on port $port is not responding"
    fi
done

# Check RL coordinator
if curl -f -s "http://localhost:5000/health" > /dev/null; then
    echo "✅ RL Coordinator is running"
else
    echo "❌ RL Coordinator is not responding"
fi

# Check Redis
if docker exec rl-redis redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis is running"
else
    echo "❌ Redis is not responding"
fi

# Check PostgreSQL
if docker exec rl-postgres pg_isready -U postgres > /dev/null 2>&1; then
    echo "✅ PostgreSQL is running"
else
    echo "❌ PostgreSQL is not responding"
fi

echo ""
echo "🎮 RL Training Environment Status:"
echo "================================="
echo "Game Servers:     http://localhost:8081, 8082, 8083"
echo "RL Coordinator:   http://localhost:5000"
echo "TensorBoard:      http://localhost:6006"
echo "Database:         localhost:5432"
echo ""
echo "📊 To monitor training:"
echo "  - View logs: docker-compose -f docker-compose.rl.yml logs -f rl-coordinator"
echo "  - TensorBoard: http://localhost:6006"
echo "  - API Status: curl http://localhost:5000/stats"
echo ""
echo "⚡ To start training:"
echo "  docker exec -it rl-coordinator python coordinator.py"
echo ""
echo "🛑 To stop everything:"
echo "  docker-compose -f docker-compose.rl.yml down"
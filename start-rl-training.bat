@echo off
echo 🚀 Starting Terraforming Mars RL Training Environment
echo ==================================================

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not running. Please start Docker first.
    pause
    exit /b 1
)

REM Create necessary directories
if not exist "rl-models" mkdir rl-models
if not exist "rl-logs" mkdir rl-logs

REM Stop any existing containers
echo 🧹 Cleaning up existing containers...
docker-compose -f docker-compose.rl.yml down

REM Build and start the RL environment
echo 🏗️ Building and starting RL environment...
docker-compose -f docker-compose.rl.yml up --build -d

REM Wait for services to start
echo ⏳ Waiting for services to start...
timeout /t 30 /nobreak >nul

REM Check service health
echo 🔍 Checking service health...

REM Check game servers
for %%p in (8081 8082 8083) do (
    curl -f -s "http://localhost:%%p/" >nul 2>&1
    if errorlevel 1 (
        echo ❌ Game server on port %%p is not responding
    ) else (
        echo ✅ Game server on port %%p is running
    )
)

REM Check RL coordinator
curl -f -s "http://localhost:5000/health" >nul 2>&1
if errorlevel 1 (
    echo ❌ RL Coordinator is not responding
) else (
    echo ✅ RL Coordinator is running
)

REM Check Redis
docker exec rl-redis redis-cli ping >nul 2>&1
if errorlevel 1 (
    echo ❌ Redis is not responding
) else (
    echo ✅ Redis is running
)

REM Check PostgreSQL
docker exec rl-postgres pg_isready -U postgres >nul 2>&1
if errorlevel 1 (
    echo ❌ PostgreSQL is not responding
) else (
    echo ✅ PostgreSQL is running
)

echo.
echo 🎮 RL Training Environment Status:
echo =================================
echo Game Servers:     http://localhost:8081, 8082, 8083
echo RL Coordinator:   http://localhost:5000
echo TensorBoard:      http://localhost:6006
echo Database:         localhost:5432
echo.
echo 📊 To monitor training:
echo   - View logs: docker-compose -f docker-compose.rl.yml logs -f rl-coordinator
echo   - TensorBoard: http://localhost:6006
echo   - API Status: curl http://localhost:5000/stats
echo.
echo ⚡ To start training:
echo   docker exec -it rl-coordinator python coordinator.py
echo.
echo 🛑 To stop everything:
echo   docker-compose -f docker-compose.rl.yml down
echo.
echo Press any key to open the dashboard in your browser...
pause >nul
start http://localhost:5000/dashboard
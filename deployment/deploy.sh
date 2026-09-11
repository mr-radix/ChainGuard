#!/usr/bin/env bash
# ==============================================================================
# ChainGuard Deployment & Multi-Pass Automated Verification Pipeline
# ==============================================================================

set -e

GREEN='\033[0;32m'
CYAN='\033[0;36m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${CYAN}========================================================================${NC}"
echo -e "${CYAN}          CHAINGUARD MULTI-PASS DEPLOYMENT VERIFICATION ENGINE          ${NC}"
echo -e "${CYAN}========================================================================${NC}"

# Pass 1: Python Environment & Version Checks
echo -e "\n${YELLOW}[PASS 1/5] Verifying Model & Environment Dependencies...${NC}"
python3 -c "import torch, xgboost, sklearn, pandas, numpy, networkx; print(f'PyTorch: {torch.__version__}, XGBoost: {xgboost.__version__}, Sklearn: {sklearn.__version__}')"
echo -e "${GREEN}✓ All pinned model dependencies verified.${NC}"

# Pass 2: Unit Testing Suite (Pytest)
echo -e "\n${YELLOW}[PASS 2/5] Running Backend Unit Tests (Pytest Suite)...${NC}"
(cd model && python3 -m pytest tests/ -v --tb=short)
echo -e "${GREEN}✓ 100% Pytest test suite passed.${NC}"

# Pass 3: Model Export & Model Registry Initialization
echo -e "\n${YELLOW}[PASS 3/5] Testing Model Export & Production Registry Initialization...${NC}"
(cd model && python3 serve_models.py --export-only --data-dir Dataset)
if [ -f "model/exported_models/model_registry.json" ]; then
    echo -e "${GREEN}✓ Model registry and compiled artifacts created successfully.${NC}"
else
    echo -e "${RED}✗ Model registry export failed!${NC}"
    exit 1
fi

# Pass 4: Model REST Server Smoke Test & Endpoint Verification
echo -e "\n${YELLOW}[PASS 4/5] Launching Test REST API Server & Testing Live Endpoints...${NC}"
TEST_PORT=8089
(cd model && python3 serve_models.py --serve --port $TEST_PORT) &
SERVER_PID=$!


# Wait for server startup
sleep 3

echo -e "Testing GET /health..."
curl -s -f http://localhost:$TEST_PORT/health | grep -q "healthy"
echo -e "${GREEN}✓ /health returned healthy status.${NC}"

echo -e "Testing GET /models..."
curl -s -f http://localhost:$TEST_PORT/models | grep -q "tabular_models"
echo -e "${GREEN}✓ /models returned model registry.${NC}"

echo -e "Testing POST /predict/ransomware..."
curl -s -f -X POST http://localhost:$TEST_PORT/predict/ransomware \
    -H "Content-Type: application/json" \
    -d '{"address":"13AM4VW2dhxYgXeQepoHkHSQuy6NgaEb94","income":2500000000,"loop":4,"count":42,"length":10,"weight":0.5,"neighbors":8}' \
    | grep -q "ransomware_risk_score"
echo -e "${GREEN}✓ /predict/ransomware prediction successful.${NC}"

echo -e "Testing POST /predict/risk..."
curl -s -f -X POST http://localhost:$TEST_PORT/predict/risk \
    -H "Content-Type: application/json" \
    -d '{"address":"13AM4VW2dhxYgXeQepoHkHSQuy6NgaEb94","income":2500000000,"loop":4,"count":42,"length":10,"weight":0.5,"neighbors":8}' \
    | grep -q "risk_assessment"
echo -e "${GREEN}✓ /predict/risk composite evaluation successful.${NC}"

echo -e "Testing Address Format Validation (HTTP 400 Error)..."
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:$TEST_PORT/predict/risk \
    -H "Content-Type: application/json" \
    -d '{"address":"john","blockchain":"BTC"}')
if [ "$HTTP_STATUS" -eq 400 ]; then
    echo -e "${GREEN}✓ Invalid address ('john') correctly rejected with HTTP 400 Bad Request.${NC}"
else
    echo -e "${RED}✗ Address validation check failed (Expected 400, got $HTTP_STATUS)${NC}"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

# Stop test server cleanly
set +e
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
set -e



# Pass 5: Website Dashboard Integrity & Asset Verification
echo -e "\n${YELLOW}[PASS 5/5] Verifying Website Dashboard Integrity & Assets...${NC}"
if [ -f "website/src/index.html" ] && [ -f "website/src/styles.css" ] && [ -f "website/src/app.js" ]; then
    echo -e "${GREEN}✓ Website HTML, CSS, and JS assets verified.${NC}"
else
    echo -e "${RED}✗ Website dashboard files missing!${NC}"
    exit 1
fi

echo -e "\n${CYAN}========================================================================${NC}"
echo -e "${GREEN}       🎉 MULTI-PASS VERIFICATION COMPLETE: DEPLOYMENT READY! 🎉         ${NC}"
echo -e "${CYAN}========================================================================${NC}"
echo -e "To launch in production using Docker Compose:"
echo -e "  cd deployment && docker-compose up --build"
echo -e "\nTo launch manually:"
echo -e "  Model REST Server:   python3 deployment/model/serve_models.py --serve --port 8080"
echo -e "  Website Dashboard:  cd deployment/website && npm start"
echo -e "${CYAN}========================================================================${NC}"

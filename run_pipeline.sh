#!/bin/bash
# ─────────────────────────────────────────────────────────────
#  ML Surrogate Model for Sheet Metal Forming
#  Full Pipeline — Run once to execute everything
#
#  Usage:
#    bash run_pipeline.sh          # full pipeline
#    bash run_pipeline.sh --no-rag # skip RAG (no API key needed)
# ─────────────────────────────────────────────────────────────

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

RUN_RAG=true
if [[ "$1" == "--no-rag" ]]; then
  RUN_RAG=false
fi

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}   ML Surrogate Model — Sheet Metal Forming         ${NC}"
echo -e "${BLUE}   FEM × PyTorch × SHAP × PSO × RAG                ${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════${NC}"
echo ""

# ── Check if real metadata.csv exists ─────────────────────────
if [ -f "data/metadata.csv" ]; then
  echo -e "  ${GREEN}✅ Real metadata.csv found — using real DDACS data${NC}"
else
  echo -e "  ${YELLOW}⚠️  No metadata.csv found in data/             ${NC}"
  echo -e "  ${YELLOW}   Drop your metadata.csv into data/ folder     ${NC}"
  echo -e "  ${YELLOW}   Using synthetic data for now...               ${NC}"
fi
echo ""

# ── Step 0: Install dependencies ──────────────────────────────
echo -e "${YELLOW}[0/6] Installing dependencies...${NC}"
pip3 install torch numpy pandas matplotlib scikit-learn shap \
             sentence-transformers faiss-cpu --quiet
if [ "$RUN_RAG" = true ]; then
  pip3 install groq anthropic --quiet 2>/dev/null || true
fi
echo -e "${GREEN}    ✅ Dependencies installed${NC}"
echo ""

# ── Step 1: Data ───────────────────────────────────────────────
echo -e "${YELLOW}[1/6] Loading / generating dataset...${NC}"
python3 1_generate_data.py
echo -e "${GREEN}    ✅ Data ready${NC}"
echo ""

# ── Step 2: Surrogate Model ────────────────────────────────────
echo -e "${YELLOW}[2/6] Training PyTorch surrogate model...${NC}"
python3 2_surrogate_model.py
echo -e "${GREEN}    ✅ Model trained${NC}"
echo ""

# ── Step 3: SHAP ───────────────────────────────────────────────
echo -e "${YELLOW}[3/6] Running SHAP explainability analysis...${NC}"
python3 3_shap_analysis.py
echo -e "${GREEN}    ✅ SHAP complete${NC}"
echo ""

# ── Step 4: PSO ────────────────────────────────────────────────
echo -e "${YELLOW}[4/6] Running PSO process optimisation...${NC}"
python3 4_pso_optimisation.py
echo -e "${GREEN}    ✅ Optimisation complete${NC}"
echo ""

# ── Step 5: Report ─────────────────────────────────────────────
echo -e "${YELLOW}[5/6] Generating full summary report...${NC}"
python3 5_full_report.py
echo -e "${GREEN}    ✅ Report generated${NC}"
echo ""

# ── Step 6: RAG ────────────────────────────────────────────────
if [ "$RUN_RAG" = true ]; then
  echo -e "${YELLOW}[6/6] Building RAG knowledge assistant...${NC}"
  echo -e "      ${CYAN}(runs demo queries then opens interactive mode)${NC}"
  python3 6_rag_assistant.py
  echo -e "${GREEN}    ✅ RAG assistant ready${NC}"
else
  echo -e "${YELLOW}[6/6] Skipping RAG (--no-rag flag set)${NC}"
fi
echo ""

# ── Done ──────────────────────────────────────────────────────
echo -e "${BLUE}════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}   ✅ FULL PIPELINE COMPLETE!${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  📁 All results saved in: ${YELLOW}./results/${NC}"
echo ""
echo -e "  ${GREEN}→${NC} results/1_data_distributions.png"
echo -e "  ${GREEN}→${NC} results/2_surrogate_model_results.png"
echo -e "  ${GREEN}→${NC} results/3a_shap_regression_importance.png"
echo -e "  ${GREEN}→${NC} results/3b_shap_crack_analysis.png"
echo -e "  ${GREEN}→${NC} results/3c_shap_heatmap_all_outcomes.png"
echo -e "  ${GREEN}→${NC} results/4_pso_optimisation_results.png"
echo -e "  ${GREEN}→${NC} results/5_full_project_summary.png"
echo -e "  ${GREEN}→${NC} results/optimal_parameters.csv"
echo -e "  ${GREEN}→${NC} results/rag_qa_log.csv"
echo -e "  ${GREEN}→${NC} rag_db/forming_index.faiss"
echo ""
echo -e "  To add your real data: copy metadata.csv into ${YELLOW}data/${NC}"
echo -e "  To use LLM in RAG: set ${YELLOW}GROQ_API_KEY${NC} env variable"
echo ""
open results/ 2>/dev/null || true

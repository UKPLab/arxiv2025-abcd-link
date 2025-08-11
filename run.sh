#!/bin/bash

set -e
set -o pipefail

# --------------------------
# Parse flags
# --------------------------
RUN_RETRIEVAL=false
RUN_PROMPTS=false
RUN_LLM=false
RUN_EVAL=false
RUN_ANNO=false
RUN_GOLD=false

if [ "$#" -eq 0 ]; then
  # Default: run everything
  RUN_RETRIEVAL=true
  RUN_PROMPTS=true
  RUN_LLM=true
  RUN_EVAL=true
  RUN_ANNO=true
  RUN_GOLD=true
else
  for arg in "$@"; do
    case $arg in
      --retrieval) RUN_RETRIEVAL=true ;;
      --prompts) RUN_PROMPTS=true ;;
      --llm) RUN_LLM=true ;;
      --eval) RUN_EVAL=true ;;
      --anno) RUN_ANNO=true ;;
      --gold) RUN_GOLD=true ;;
      *) echo "Unknown option: $arg" && exit 1 ;;
    esac
  done
fi

# --------------------------
# Step 1: Retrieval
# --------------------------
if [ "$RUN_RETRIEVAL" = true ]; then
  echo "[Step 1] Running retrieval models..."
  python scripts/run_retrievals.py
  echo "Retrieval complete."
fi

# --------------------------
# Step 2: Prompt generation
# --------------------------
if [ "$RUN_PROMPTS" = true ]; then
  echo "[Step 2] Generating prompts..."
  python prompts/generate_prompts.py
  echo "Prompts generated."
fi

# --------------------------
# Step 3: LLM inference
# --------------------------
if [ "$RUN_LLM" = true ]; then
  echo "[Step 3] Running LLM inference..."
  python scripts/run_llm_inference.py
  echo "LLM inference complete."
fi

# --------------------------
# Step 4: Evaluation
# --------------------------
if [ "$RUN_EVAL" = true ]; then
  echo "[Step 4] Evaluating results..."
  python scripts/evaluate.py
  echo "Evaluation complete. See: ./eval_outputs/"
fi

# --------------------------
# Step 5: Human Annotation Analysis
# --------------------------
if [ "$RUN_ANNO" = true ]; then
  echo "[Step 5] Generating acceptance rates from annotations..."
  python scripts/annotation_results.py
  echo "Annotation evaluation complete. See: ./datasets/*_he"
fi

# --------------------------
# Step 6: Gold Label Evaluation
# --------------------------
if [ "$RUN_GOLD" = true ]; then
  echo "[Step 6] Evaluating on gold label annotations..."
  python scripts/evaluate_gold_labels.py
  echo "Gold label evaluation complete. See: ./datasets/*_he/eval_gold_labels.json"
fi

echo "All requested steps completed successfully."
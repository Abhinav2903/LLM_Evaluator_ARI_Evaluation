# LLM_Evaluator_ARI_Evaluation

Reliability-first evaluation pipeline for **LLM-as-a-Judge** under **readability control (ARI)**.  
Includes single-judge and ensemble (PoLL) evaluators, Baseline/CoT/JARI prompts, and reliability metrics (CR, RR, PBR).

---

## 🔑 Requirements

- **Python 3.10+**
- **pip**
- **OpenAI API key** (required for judge/evaluator runs)
- (Optional) **7-Zip** CLI (`7z`) or any archiver to extract `experiment.7z`

---

## 🚀 Quickstart
# 1) (Recommended) create & activate a virtual environment
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# 2) Install dependencies
pip install -r requirements.txt

.
├─ data/         # ✅ Ready-to-use ARI-controlled dataset (already included)
├─ reports/            # 📊 All experiment/evaluation reports & figures
├─ experiment.7z       # 🧪 End-to-end experiments (curation → LLM-judge)
├─ requirements.txt
└─ README.md


@mastersthesis{pareek2025llmari,
  title  = {Evaluating LLM Evaluator Reliability Across Readability Levels and the Impact of Ensemble Approaches},
  author = {Pareek, Abhinav},
  school = {Otto-von-Guericke-Universität Magdeburg},
  year   = {2025}
}

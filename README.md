```markdown
# EcoRec: Ecology-Aware Recommendation System  
> **A lightweight, production-ready recommendation engine that predicts "ecological link formation" (i.e., which items can *drive* others) using topological features + content similarity.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688)](https://fastapi.tiangolo.com/)
[![SQLite](https://img.shields.io/badge/SQLite-3-lightgrey)](https://www.sqlite.org/)

##  What Makes EcoRec Unique?

| Traditional RecSys | EcoRec |
|-------------------|--------|
| ❌ Recommends only popular/high-score items | ✅ Recommends **ecosystem drivers** (high hub + high out-degree) |
| ❌ Black-box models (e.g., DNN) | ✅ **White-box link prediction** with business-aligned scores |
| ❌ No cold-start solution | ✅ **Content-based fallback**: TF-IDF + fuzzy matching |
| ❌ Offline evaluation only | ✅ **End-to-end A/B testing**: `/ab/stats` + click/conversion logging |

### Core Innovation: **Link Prediction for Ecosystem Health**
We model items as a graph and predict new *drive links* (A → B means "A can drive sales of B"), using:
- **Topological features**: `hub_score`, `out_degree`, `drive_efficiency`
- **Content similarity**: TF-IDF + fuzzy title matching
- **User behavior**: Collaborative filtering (lightweight GNN-inspired)

> **Key Insight**: *"In e-commerce, recommending ecosystem drivers (not just stars) improves long-term platform health."*

---

## Quick Start

### 1. Prepare Data
Place your datasets in `./data/`:
```
data/
├── 商品-社团映射(1).csv       # ASIN ↔ community mapping
├── 社团-商品列表(1).json      # Community → ASIN list
├── 10K_amazon-meta.txt        # Item metadata (title, rating, etc.)
└── node_features_10k_topology.csv  # Graph features (Pagerank, Hub, etc.)
```

### 2. Install Dependencies
```bash
pip install fastapi uvicorn pandas scikit-learn networkx matplotlib seaborn fuzzywuzzy
```

### 3. Run the Service
```bash
python api.py
# ✅ Server starts at http://localhost:8000
# ✅ Check /health and /ready endpoints
```

### 4. Test End-to-End
```bash
python simulate_click.py
# Output:
#   推荐结果: [B001, B002, ...]
#   点击 B001 → 成功
#   转化 B001 → 成功
#    A/B 测试: content_like CTR=20.2%, core_drive CTR=23.2%, hybrid CTR=26.2%
```

---

## Key Results (250 Exposures)
| Strategy | CTR | CVR | Key Insight |
|----------|-----|-----|-------------|
| `hybrid` | 26.2% | 31.2% | **Best balance**: 60% user collab + 40% drive-type |
| `core_drive` | 23.2% | 25.0% | **Eco-value proven**: High-hub items drive clicks *and* conversions |
| `content_like` | 20.2% | 14.3% | Pure similarity → low-quality "ghost items" hurt CVR |

> **Finding**: A high-drive item (`891061320`) was clicked & converted **twice**, proving ecosystem drivers have *sustainable commercial value*.

---

## Project Structure
| File | Role |
|------|------|
| `main.py` | Core algorithms: link prediction, collaborative filtering, strategy scheduler |
| `api.py` | FastAPI service gateway (with health checks & fallbacks) |
| `log_db.py` | SQLite-based logging & A/B test framework (✅ fixes SQLite `ORDER BY` bug) |
| `simulate_click.py` | Automated end-to-end test simulator |
| `data/` | Datasets (item metadata, topological features, etc.) |

---

## 📚Citation
If you use EcoRec in research, please cite:
```bibtex
@misc{ecorec2026,
  author = {lindvalllowrie},
  title = {EcoRec: Ecology-Aware Recommendation via Topological Link Prediction},
  year = {2026},
  howpublished = {\url{https://github.com/LindvallLowrie/ecorec}}
}
```

---
```
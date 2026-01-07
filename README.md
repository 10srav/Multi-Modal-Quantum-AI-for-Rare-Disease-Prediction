# Multi-Modal Quantum AI for Rare Disease Prediction

## Case Study: Hutchinson-Gilford Progeria Syndrome (HGPS)

A comprehensive AI system that combines classical machine learning with quantum machine learning (QML) for rare disease risk prediction and progression assessment.

---

## Overview

This project implements a multi-modal AI system for detecting Hutchinson-Gilford Progeria Syndrome (HGPS), an extremely rare genetic disorder causing accelerated aging in children. The system:

- **Fuses multiple data modalities**: Facial images + clinical/tabular data
- **Compares classical ML vs Quantum ML**: QSVM and QNN vs traditional models
- **Addresses data scarcity**: Optimized for extremely limited training data
- **Provides explainable predictions**: Feature importance and visual explanations
- **Generates clinical recommendations**: Confidence-calibrated risk assessments

### Key Features

- Multi-modal deep learning (CNN + MLP fusion)
- Quantum machine learning with Qiskit (QSVM, VQC/QNN)
- Synthetic data generation based on HGPS literature
- FastAPI REST backend
- Interactive Streamlit dashboard
- Comprehensive experiment framework

---

## UN Sustainable Development Goals Alignment

| SDG | Alignment |
|-----|-----------|
| **SDG 3: Good Health** | Early detection for rare disease intervention |
| **SDG 9: Innovation** | Novel quantum computing healthcare applications |
| **SDG 10: Reduced Inequalities** | Improving rare disease diagnostic access |

---

## Project Structure

```
project/
├── src/
│   ├── __init__.py
│   ├── data.py              # Data generation, preprocessing, loaders
│   ├── models.py            # Classical ML models and training
│   ├── api.py               # FastAPI REST endpoints
│   ├── dashboard.py         # Streamlit web interface
│   ├── features/
│   │   ├── face_cnn.py      # CNN for facial feature extraction
│   │   ├── tabular_mlp.py   # MLP for tabular data
│   │   └── fusion.py        # Multi-modal fusion models
│   └── qml/
│       ├── qsvm.py          # Quantum Support Vector Machine
│       ├── qnn.py           # Quantum Neural Network (VQC)
│       └── quantum_features.py  # Feature maps and ansatzes
├── notebooks/
│   └── experiments.ipynb    # Experiment and ablation studies
├── tests/
│   ├── test_data.py
│   ├── test_models.py
│   └── test_api.py
├── data/                    # Generated data directory
├── models/                  # Saved model checkpoints
├── results/                 # Experiment results
├── figures/                 # Generated plots
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.11+
- pip or conda
- (Optional) Kaggle API credentials for downloading datasets
- (Optional) CUDA for GPU acceleration

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/hgps-multimodal-ai.git
cd hgps-multimodal-ai
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **(Optional) Configure Kaggle API**:
```bash
# Place your kaggle.json in ~/.kaggle/
# Or set environment variables:
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

---

## Quick Start

### 1. Generate Synthetic Data

```python
from src.data import prepare_full_dataset

data = prepare_full_dataset(
    n_hgps=50,
    n_controls=450,
    generate_images=True
)

print(f"Total samples: {len(data['full_df'])}")
```

### 2. Train Classical Models

```python
from src.models import ClassicalTabularModels

models = ClassicalTabularModels(calibrate=True)
models.fit(X_train, y_train, X_val, y_val)
results = models.evaluate(X_test, y_test)
```

### 3. Train Quantum Models

```python
from src.qml import train_qsvm, train_qnn, evaluate_qsvm

# QSVM
qsvm = train_qsvm(X_train, y_train, num_features=6, use_quantum=True)
qsvm_results = evaluate_qsvm(qsvm, X_test, y_test)

# QNN
qnn = train_qnn(X_train, y_train, num_features=6, ansatz_reps=3)
```

### 4. Run API Server

```bash
uvicorn src.api:app --reload --port 8000
```

Visit `http://localhost:8000/docs` for interactive API documentation.

### 5. Launch Dashboard

```bash
streamlit run src/dashboard.py
```

Visit `http://localhost:8501` for the web interface.

---

## Production Deployment

### Quick Start (Docker)

```bash
# Build and run all services
docker-compose up --build

# Or run specific services
docker-compose up api dashboard
```

Services:
- **API**: http://localhost:8000 (with OpenAPI docs at /docs)
- **Dashboard**: http://localhost:8501

### Train Models First

```bash
# Train all models before deployment
python -m src.train --all

# Or use Docker
docker-compose --profile training run trainer
```

### Environment Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Key settings:
```env
ENVIRONMENT=production
API_KEY=your-secure-api-key
LOG_LEVEL=INFO
DEVICE=auto
```

### Production with Nginx (Full Stack)

```bash
docker-compose --profile production-full up
```

This includes:
- API server (4 workers)
- Streamlit dashboard
- Redis for rate limiting
- Nginx reverse proxy

### API Authentication

Include API key in requests when enabled:

```bash
curl -H "X-API-Key: your-api-key" http://localhost:8000/predict/tabular ...
```

### Health Monitoring

```bash
# Check API health
curl http://localhost:8000/health

# Check loaded models
curl http://localhost:8000/models
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | Multimodal prediction (image + clinical) |
| `/predict/tabular` | POST | Tabular-only prediction |
| `/predict/qml` | POST | Quantum vs Classical comparison |
| `/explain` | POST | Feature importance explanation |
| `/growth-curve/{age}` | GET | Growth trajectory prediction |

### Example Request

```bash
curl -X POST "http://localhost:8000/predict/tabular" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 5.0,
    "height_cm": 85.0,
    "weight_kg": 12.0,
    "small_jaw": 1,
    "prominent_eyes": 1,
    "thin_skin": 1,
    "hair_loss": 0,
    "lmna_mut": 0
  }'
```

---

## Experiments

Run the Jupyter notebook for comprehensive experiments:

```bash
jupyter notebook notebooks/experiments.ipynb
```

### Experiment Coverage

1. **Classical ML Baselines**: SVM, Random Forest, XGBoost, MLP
2. **Quantum ML Comparison**: QSVM vs Classical SVM, QNN vs MLP
3. **Data Scarcity Analysis**: Performance vs training sample size
4. **Ablation Studies**: Unimodal vs multimodal
5. **Calibration Analysis**: Probability reliability

---

## Model Architecture

### Classical Models

```
Face Image → ResNet18 → Face Embedding (256-d)
                                         ↘
                                          Fusion → MLP → Risk Score
                                         ↗
Clinical Data → MLP → Tabular Embedding (64-d)
```

### Quantum Models

```
Selected Features (6) → Quantum Feature Map (ZZ) → Quantum Kernel/Ansatz → Measurement → Prediction
```

---

## Performance Metrics

| Model | Accuracy | F1 Score | AUC |
|-------|----------|----------|-----|
| XGBoost | 0.93 | 0.87 | 0.95 |
| Random Forest | 0.91 | 0.84 | 0.93 |
| Classical SVM | 0.89 | 0.82 | 0.91 |
| **QSVM** | 0.88 | 0.80 | 0.90 |
| Multimodal Fusion | 0.94 | 0.89 | 0.96 |

*Note: Results on synthetic data. Performance may vary.*

---

## Testing

Run the test suite:

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_data.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PYTHONPATH` | Python path | `/app` |
| `MODEL_DIR` | Model checkpoint directory | `models/` |
| `DATA_DIR` | Data directory | `data/` |

### Model Configuration

Edit model parameters in `src/models.py` or pass to constructors:

```python
model = FaceCNN(
    embedding_dim=256,
    pretrained=True,
    freeze_backbone=True,
    dropout=0.3
)
```

---

## Limitations & Disclaimer

**This is a research and educational project.**

- Uses synthetic data; not validated on real clinical data
- Not intended for actual clinical diagnosis
- QML experiments run on simulators, not quantum hardware
- Performance claims are on synthetic benchmarks only

**Always consult qualified healthcare professionals for medical decisions.**

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push to branch: `git push origin feature/new-feature`
5. Submit a Pull Request

---

## License

MIT License - see LICENSE file for details.

---

## Citation

If you use this project in your research, please cite:

```bibtex
@software{hgps_multimodal_ai,
  title={Multi-Modal Quantum AI for Rare Disease Prediction},
  author={HGPS-AI Research Team},
  year={2024},
  url={https://github.com/yourusername/hgps-multimodal-ai}
}
```

---

## Acknowledgments

- HGPS medical literature for clinical parameter guidance
- Qiskit team for quantum computing framework
- Streamlit and FastAPI communities

---

## Contact

For questions or collaboration inquiries, please open an issue on GitHub.

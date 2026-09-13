# AAMCF

**Adaptive Agent Memory Compression Framework**

AAMCF is a memory-management pipeline for LLM agents that decides **what information is worth remembering, how useful it is, and whether to store, compress, archive, or delete it.**

## Pipeline

```text
Conversation
     ↓
Stage 0a: Candidate Gate
     ↓
Stage 0b: Clause Segmentation
     ↓
Stage 0c: GLiNER Extraction
     ↓
Stage 0d–0f: Validation + Deduplication
     ↓
Stage 1: Feature Extraction
     ↓
Stage 2: Utility Prediction
     ↓
Stage 3: Calibration
     ↓
Stage 4: Memory Decision
     ↓
Stage 5: SQLite + FAISS
     ↓
Stage 6: Retrieval
```

---

## Tech Stack

* Python 3.12
* PyTorch
* Hugging Face Transformers
* DistilBERT
* GLiNER
* LightGBM
* Scikit-learn
* SQLite
* FAISS
* Sentence Transformers

---

## Setup

Using `uv`:

```bash
uv venv
```

Windows:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
uv pip install -r requirements.txt
```

---

# 1. Dataset Preparation

AAMCF uses:

* **LoCoMo**
* **LongMemEval-S**

Place them under:

```text
data/
├── locomo/
└── longmemeval/
    └── longmemeval_s_cleaned.json
```

The datasets are converted into a common internal format:

```python
{
    "conversation_id": "...",
    "turn_id": "...",
    "text": "...",
    "timestamp": "..."
}
```

---

# 2. Stage 0 — Candidate Memory Extraction

### 0a. Candidate Gate

A DistilBERT binary classifier filters irrelevant turns.

```text
Input turn
    ↓
DistilBERT
    ↓
P(candidate)
    ↓
candidate / discard
```

Model:

```python
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)

MODEL = "distilbert-base-uncased"

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL)

model = DistilBertForSequenceClassification.from_pretrained(
    MODEL,
    num_labels=2
)
```

Inference:

```python
def is_candidate(text, threshold=0.30):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        logits = model(**inputs).logits

    probability = torch.softmax(
        logits,
        dim=-1
    )[0, 1].item()

    return probability >= threshold
```

The threshold is selected using the validation set rather than assuming `0.5`.

---

### 0b. Clause Segmentation

Candidate turns are split into atomic clauses.

```text
"I live in Mumbai and I prefer Python."

        ↓

"I live in Mumbai."
"I prefer Python."
```

---

### 0c. GLiNER Extraction

GLiNER extracts structured entities/slots from each clause.

```python
entities = gliner_model.predict_entities(
    clause,
    labels
)
```

---

### 0d–0f. Validation + Deduplication

Invalid and duplicate candidates are removed before feature extraction.

Final candidate:

```python
{
    "memory_id": "...",
    "conversation_id": "...",
    "turn_id": "...",
    "text": "...",
    "entity_type": "...",
    "confidence": 0.91
}
```

---

# 3. Stage 1 — Feature Extraction

Each memory is converted into an **8-dimensional feature vector**.

```python
FEATURES = [
    "semantic_relevance",
    "temporal_relevance",
    "recurrence",
    "explicitness",
    "entity_specificity",
    "retrieval_frequency",
    "contradiction_risk",
    "staleness"
]
```

Example:

```python
X = [
    memory["semantic_relevance"],
    memory["temporal_relevance"],
    memory["recurrence"],
    memory["explicitness"],
    memory["entity_specificity"],
    memory["retrieval_frequency"],
    memory["contradiction_risk"],
    memory["staleness"]
]
```

---

# 4. Stage 2 — Utility Prediction

The utility model predicts:

```text
How useful is this memory?
```

Input:

```text
8 features
```

Output:

```text
raw utility score
```

Example LightGBM model:

```python
import lightgbm as lgb

model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.03,
    num_leaves=31,
    random_state=42
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)]
)
```

---

# 5. Stage 3 — Calibration

The raw utility score is calibrated using a small gold/counterfactual set.

```python
from sklearn.isotonic import IsotonicRegression

calibrator = IsotonicRegression(
    y_min=0,
    y_max=1,
    out_of_bounds="clip"
)

calibrator.fit(
    raw_scores,
    gold_labels
)

probability = calibrator.predict(
    [raw_score]
)[0]
```

Evaluation:

```python
from sklearn.metrics import brier_score_loss

brier = brier_score_loss(
    y_true,
    probabilities
)
```

Also report **Expected Calibration Error (ECE)**.

---

# 6. Stage 4 — Memory Decision

The calibrated probability determines the memory action.

```python
def decide(p):

    if p >= 0.75:
        return "store"

    if p >= 0.50:
        return "compress"

    if p >= 0.25:
        return "archive"

    return "delete"
```

Thresholds should be tuned experimentally.

---

# 7. Stage 5 — Memory Storage

AAMCF uses:

```text
SQLite → metadata
FAISS  → semantic search
```

SQLite example:

```sql
CREATE TABLE memories (
    memory_id TEXT PRIMARY KEY,
    conversation_id TEXT,
    text TEXT,
    utility REAL,
    probability REAL,
    status TEXT,
    created_at TEXT
);
```

Embeddings:

```python
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

embedding = embedder.encode(
    memory_text
)
```

---

# 8. Stage 6 — Retrieval

```python
def retrieve(query, k=5):

    query_embedding = embedder.encode(
        [query]
    )

    scores, indices = index.search(
        query_embedding,
        k
    )

    return [
        memories[i]
        for i in indices[0]
    ]
```

Retrieved memories are passed into the agent context.

---

# 9. Training Labels

The system can use multiple supervision sources:

```text
LoCoMo evidence
LongMemEval evidence
LLM-labelled data
Token-overlap proxy
Counterfactual ablation
```

For the counterfactual gold signal:

```python
Y = max(
    0,
    min(
        1,
        accuracy_full - accuracy_without_memory
    )
)
```

The counterfactual set is kept small because it requires running the QA evaluation twice.

---

# 10. Evaluation

Evaluate the system at three levels.

### Stage 0a

```text
Precision
Recall
F1
```

Recall is especially important because incorrectly rejecting a useful turn prevents it from reaching the rest of the memory pipeline.

### Stage 2

```text
MAE
RMSE
Spearman correlation
```

### Stage 3

```text
Brier Score
ECE
Reliability Diagram
```

### End-to-End

Compare:

```text
Full-history baseline
No-memory baseline
AAMCF
```

Measure:

```text
QA accuracy
Memory count
Storage reduction
Retrieval quality
Latency
```

---

# 11. Running the Pipeline

Prepare datasets:

```bash
python scripts/prepare_locomo.py
python scripts/prepare_longmemeval.py
```

Build Stage 0a dataset:

```bash
python scripts/build_stage0a_dataset.py
```

Train Stage 0a:

```bash
python scripts/train_gate.py
```

Build Stage 2 dataset:

```bash
python scripts/build_stage2_dataset.py
```

Train utility model:

```bash
python scripts/train_utility.py
```

Run evaluation:

```bash
python scripts/evaluate.py
```

Run the complete pipeline:

```bash
python src/pipeline.py
```

---

# 12. Reproducibility

Use a fixed seed:

```python
SEED = 42
```

and split data by:

```text
conversation/session
```

rather than individual turns to prevent information leakage between train and test sets.

---

# 13. Output

A processed memory should ultimately look like:

```json
{
    "memory_id": "M102",
    "text": "User prefers Python for backend development.",
    "utility": 0.78,
    "calibrated_probability": 0.84,
    "decision": "store"
}
```

The final memory store therefore contains only information that survives the complete:

```text
Gate
→ Extract
→ Feature
→ Predict
→ Calibrate
→ Decide
→ Store
```

pipeline.

# Comparison Analysis: RNA Folding vs 0.365 RNAPro Inference

## Key Finding: ⚠️ **Neither notebook actually uses RNAPro!**

Despite the filename `0-365-rnapro-inference-with-tbm-tune.ipynb` suggesting RNAPro usage, **both notebooks only use Template-Based Modeling (TBM)**. There is no RNAPro model loading, inference, or deep learning code present.

---

## Code Comparison

### Similarities (Both notebooks are nearly identical)

Both notebooks use the **same TBM approach**:

1. **Data Loading**: Identical - loads train/test/validation sequences and labels
2. **Core Functions**: Same structure:
   - `find_similar_sequences()` - Finds similar sequences using pairwise alignment
   - `adapt_template_to_query()` - Maps template coordinates to query sequence
   - `adaptive_rna_constraints()` - Applies geometric constraints
   - `generate_rna_structure()` - Fallback structure generation
   - `predict_rna_structures()` - Main prediction function

### Differences (Minor parameter tuning)

| Feature | RNA FOLDING PREDICTION #27JAN | 0-365-rnapro-inference-with-tbm-tune |
|---------|------------------------------|--------------------------------------|
| **Constraint Strength** | `0.5 * (1.0 - min(confidence, 0.9))` | `0.4 * (1.0 - min(confidence, 0.9))` |
| **Random Noise Scale** | `max(0.05, (0.5 - similarity) * 0.25)` | `max(0.05, (0.5 - similarity) * 0.15)` |
| **Comments** | English | Russian/English mix |
| **Score** | Not specified | Claims 0.365 score |

### Detailed Code Differences

#### 1. `adaptive_rna_constraints()` function

**RNA FOLDING PREDICTION #27JAN:**
```python
constraint_strength = 0.5 * (1.0 - min(confidence, 0.9))
```

**0-365-rnapro-inference-with-tbm-tune:**
```python
constraint_strength = 0.4 * (1.0 - min(confidence, 0.9))
```
- **Impact**: Lower constraint strength (0.4 vs 0.5) means less aggressive geometric refinement

#### 2. `predict_rna_structures()` function - Random noise

**RNA FOLDING PREDICTION #27JAN:**
```python
random_scale = max(0.05, (0.5 - similarity) * 0.25)
```

**0-365-rnapro-inference-with-tbm-tune:**
```python
random_scale = max(0.05, (0.5 - similarity) * 0.15)
```
- **Impact**: Less random noise added (0.15 vs 0.25 multiplier), potentially more stable predictions

#### 3. `generate_rna_structure()` function

**RNA FOLDING PREDICTION #27JAN:**
```python
step_size = random.uniform(3.0, 5.0)
```

**0-365-rnapro-inference-with-tbm-tune:**
```python
# УХУДШЕНИЕ 6: Более случайный шаг
step_size = random.uniform(3.0, 5.0)
```
- **Note**: Same code, but 0.365 version has a comment suggesting intentional degradation ("УХУДШЕНИЕ 6" = "DEGRADATION 6")

---

## What's Missing: RNAPro Implementation

### What RNAPro Should Include

To actually use RNAPro, the code would need:

1. **Model Loading**:
```python
from rnapro import RNAProModel
model = RNAProModel.from_pretrained("nvidia/RNAPro-Private-Best-500M")
```

2. **Inference Code**:
```python
# RNAPro inference
predictions = model.predict(
    sequence=sequence,
    templates=tbm_templates,  # Optional: use TBM templates
    num_samples=5
)
```

3. **Template Format Conversion** (for hybrid approach):
```python
# Convert TBM predictions to RNAPro template format
rnapro_templates = convert_tbm_to_rnapro_format(tbm_predictions)
```

### Current Implementation: Pure TBM Only

Both notebooks use **only** these TBM steps:

1. ✅ Find similar sequences (pairwise alignment)
2. ✅ Extract template coordinates from training data
3. ✅ Adapt templates to query sequence (sequence alignment)
4. ✅ Apply geometric constraints (bond length refinement)
5. ✅ Add random noise for diversity
6. ❌ **NO RNAPro model loading**
7. ❌ **NO RNAPro inference**
8. ❌ **NO Deep learning predictions**

---

## Why the Confusion?

The filename `0-365-rnapro-inference-with-tbm-tune.ipynb` suggests:
- It should use RNAPro for inference
- It should tune TBM parameters
- It achieved a 0.365 score

**Reality**: 
- It's a pure TBM approach with slightly different parameters
- The "tune" refers to parameter tuning (constraint strength, noise scale)
- No RNAPro is actually used

---

## Summary

| Aspect | RNA FOLDING #27JAN | 0-365-rnapro-inference |
|--------|-------------------|------------------------|
| **Method** | Pure TBM | Pure TBM |
| **RNAPro Usage** | ❌ None | ❌ None (despite filename) |
| **Constraint Strength** | 0.5 | 0.4 (weaker) |
| **Noise Scale** | 0.25 | 0.15 (less noise) |
| **Complexity** | Simple TBM | Simple TBM (tuned) |
| **Score Claim** | None | 0.365 |

---

## Actual RNAPro Implementation Examples

### 1. `rnapro-inference-with-tbm-87198f.ipynb` (True Hybrid)

This notebook **actually uses RNAPro** in a hybrid approach:

**Step 1: TBM Generation** (Cells 9-11)
- Generates 5 template-based predictions using sequence alignment
- Saves to `submission_tbm.csv`

**Step 2: Convert TBM to RNAPro Templates** (Cell 16)
```python
!python preprocess/convert_templates_to_pt_files.py \
    --input_csv /kaggle/working/submission_tbm.csv \
    --output_name templates.pt
```

**Step 3: RNAPro Inference** (Cells 22-24)
- Loads RNAPro model: `self.model = RNAPro(self.configs).to(self.device)`
- Loads checkpoint: `checkpoint = torch.load(checkpoint_path, self.device)`
- Runs inference: `prediction, _, _ = self.model(...)`
- Uses TBM templates: `--template_data "./release_data/kaggle/templates.pt"`

**Step 4: Fallback for Long Sequences** (Cell 29)
- For sequences >1000 nucleotides, uses TBM predictions instead of RNAPro

### 2. `stanford-rna-3d-folding-pt2-rnapro-inference.ipynb` (Pure RNAPro)

This notebook uses **pure RNAPro**:
- Loads RNAPro model and checkpoint
- Can use precomputed templates (optional)
- Uses MSA (Multiple Sequence Alignment) data
- No TBM step - pure deep learning approach

### Key Code Differences: TBM vs RNAPro

**TBM Only** (what `0-365-rnapro-inference-with-tbm-tune.ipynb` does):
```python
# Sequence alignment
alignments = pairwise2.align.globalms(query_seq, template_seq, ...)
# Template adaptation
adapted = adapt_template_to_query(sequence, template_seq, template_coords)
# Geometric constraints
refined = adaptive_rna_constraints(adapted, sequence, confidence=similarity)
```

**RNAPro** (what `rnapro-inference-with-tbm-87198f.ipynb` does):
```python
# Model initialization
from rnapro.model.RNAPro import RNAPro
self.model = RNAPro(self.configs).to(self.device)
self.model.load_state_dict(checkpoint["model"])

# Inference
prediction, _, _ = self.model(
    input_feature_dict=data["input_feature_dict"],
    label_full_dict=None,
    label_dict=None,
    mode="inference",
)
```

## Recommendation

If you want to actually use RNAPro, you would need to:

1. **Check other notebooks**: 
   - `rnapro-inference-with-tbm-87198f.ipynb` - Shows hybrid TBM + RNAPro approach
   - `stanford-rna-3d-folding-pt2-rnapro-inference.ipynb` - Shows pure RNAPro approach

2. **Add RNAPro code**: Import and use the RNAPro model library to perform actual deep learning inference

3. **Hybrid approach**: Combine TBM (for initial templates) with RNAPro (for refinement) as shown in `rnapro-inference-with-tbm-87198f.ipynb`

**Conclusion**: The current `0-365-rnapro-inference-with-tbm-tune.ipynb` notebook is **misleadingly named** - it's a TBM-only approach with parameter tuning, not an RNAPro inference notebook. The filename suggests RNAPro usage, but the code contains no RNAPro model loading, inference, or deep learning components.

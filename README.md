# RNA Stanford Notebooks Repository

This repository contains notebooks for RNA 3D structure prediction developed for the Stanford RNA 3D Folding competition. The notebooks implement various approaches ranging from template-based modeling to deep learning methods.

## Important: RNAPro Usage Clarification

**Which notebooks actually use RNAPro?**

| Notebook | Uses RNAPro? | Method | Status |
|----------|-------------|--------|--------|
| `stanford-rna-3d-folding-pt2-rnapro-inference.ipynb` | **YES** | Pure RNAPro (deep learning) | State-of-the-art |
| `rnapro-inference-with-tbm-87198f.ipynb` | **YES** | Hybrid TBM + RNAPro | Best of both worlds |
| `0-365-rnapro-inference-with-tbm-tune.ipynb` | **NO** | TBM only | WARNING: Misleadingly named |
| `RNA FOLDING PREDICTION #27JAN.ipynb` | **NO** | TBM only | Baseline |
| `part1-sub-1-4-4-hybrid-final-take-a0a437.ipynb` | **NO** | Boltz-1 + DRfold2 + AlphaFold3 | Hybrid (different tools) |

**Key Finding**: Despite its filename, `0-365-rnapro-inference-with-tbm-tune.ipynb` does NOT contain any RNAPro model code. It only uses Template-Based Modeling (TBM) with sequence alignment and geometric constraints.

---

## Notebook Overview

### Deep Learning Approaches (RNAPro)

#### 1. `stanford-rna-3d-folding-pt2-rnapro-inference.ipynb`
**Approach**: Pure RNAPro deep learning model  
**Status**: State-of-the-art deep learning approach  
**Features**:
- Uses RNAPro-Private-Best-500M model from NVIDIA
- Can use precomputed templates (optional)
- Uses MSA (Multiple Sequence Alignment) data
- Diffusion-based sampling for structure generation
- Requires GPU resources

**Key Components**:
- Model loading: `RNAPro(self.configs)`
- Checkpoint: `rnapro-private-best-500m.ckpt`
- Inference: Deep learning forward pass
- Output: CIF format structures

**Reference**: Based on [RNAPro by NVIDIA](https://github.com/NVIDIA-Digital-Bio/RNAPro) and [Kaggle notebook](https://www.kaggle.com/code/theoviel/stanford-rna-3d-folding-pt2-rnapro-inference) by theoviel

#### 2. `rnapro-inference-with-tbm-87198f.ipynb`
**Approach**: Hybrid TBM + RNAPro  
**Status**: Combines template-based modeling with RNAPro refinement  
**Features**:
- **Step 1**: Generates TBM predictions using sequence alignment
- **Step 2**: Converts TBM predictions to RNAPro template format
- **Step 3**: RNAPro refines templates using deep learning
- **Step 4**: Fallback to TBM for sequences >1000 nucleotides

**Workflow**:
1. TBM generates 5 initial structure predictions
2. Converts TBM output to `.pt` template format
3. RNAPro uses templates to guide predictions
4. Generates 5 refined predictions (one per template)
5. For long sequences, uses pure TBM predictions

**Key Code**:
```python
# TBM step
predictions = predict_rna_structures(...)  # TBM predictions
# Convert to RNAPro format
!python preprocess/convert_templates_to_pt_files.py ...
# RNAPro inference
self.model = RNAPro(self.configs)
prediction = self.model(...)
```

---

### Template-Based Modeling (TBM) Approaches

#### 3. `0-365-rnapro-inference-with-tbm-tune.ipynb` [WARNING]
**Approach**: Template-based with tuned parameters  
**Status**: **WARNING: Misleadingly named** - Does NOT use RNAPro  
**Key Finding**: Despite filename suggesting RNAPro, this notebook only contains TBM code.

**Parameters**:
- Constraint strength: `0.4 * (1.0 - min(confidence, 0.9))`
- Random noise scale: `max(0.05, (0.5 - similarity) * 0.15)`
- Gap penalties: -3.0 (open), -0.1 (extend)
- Sequence distance range: 5.5 - 6.5 Å
- Uses deprecated `pairwise2` API

**Features**:
- Sequence alignment using BioPython `pairwise2`
- Template adaptation from training structures
- Geometric constraint refinement
- Random noise addition for diversity
- Contains Russian comments suggesting intentional parameter tuning

**What's Missing**:
- No RNAPro model loading
- No RNAPro inference code
- No deep learning components
- Only TBM (Template-Based Modeling)

#### 4. `RNA FOLDING PREDICTION #27JAN.ipynb`
**Approach**: Template-based (baseline)  
**Status**: Baseline TBM approach  
**Parameters**:
- Constraint strength: `0.5 * (1.0 - min(confidence, 0.9))`
- Random noise scale: `max(0.05, (0.5 - similarity) * 0.25)`
- Gap penalties: -3.0 (open), -0.1 (extend)
- Sequence distance range: 5.5 - 6.5 Å
- Uses deprecated `pairwise2` API

**Differences from 0.365**:
- Higher constraint strength (0.5 vs 0.4)
- More random noise (0.25 vs 0.15 multiplier)
- English comments only

---

### Other Hybrid Approaches

#### 5. `part1-sub-1-4-4-hybrid-final-take-a0a437.ipynb`
**Approach**: Hybrid (Boltz-1 + DRfold2 + AlphaFold3)  
**Status**: Production-ready hybrid approach  
**Features**:
- **Boltz-1**: Deep learning model for initial predictions
- **DRfold2**: Physics-based refinement with energy minimization
- **AlphaFold3**: Integration for long sequences (>600 nucleotides)
- **Post-processing**: Combines predictions from different methods

**Key Differences**:
- Uses external tools (not RNAPro)
- Handles long sequences specially
- Energy-based optimization instead of template matching
- More computationally intensive

**References**:
- Boltz-1: [Boltz-1 Inference Submission](https://www.kaggle.com/code/youhanlee/boltz-1-inference-submission) by youhanlee
- DRfold2: Physics-based RNA structure refinement
- AlphaFold3: DeepMind/Google DeepMind

---

## Detailed Comparison

### Method Comparison

| Feature | Pure TBM (0.365, #27JAN) | Pure RNAPro | Hybrid TBM + RNAPro | Hybrid (Boltz-1) |
|---------|-------------------------|-------------|---------------------|------------------|
| **Method** | Sequence alignment + template adaptation | Deep learning (diffusion) | TBM → RNAPro refinement | Multiple tools |
| **Input** | Sequence + training structures | Sequence (+ optional templates) | Sequence + TBM predictions | Sequence |
| **Learning** | Rule-based heuristics | Learned from data | Learned + template guidance | Learned (multiple models) |
| **Accuracy** | Good for similar sequences | State-of-the-art | Best of both worlds | High (ensemble) |
| **Speed** | Fast | Slower (neural network) | Slowest (two-stage) | Very slow |
| **Memory** | Low | High (GPU required) | Very high | Very high |
| **GPU Required** | No | Yes | Yes | Yes |

### Parameter Comparison (TBM Notebooks)

| Feature | 0.365 | #27JAN |
|---------|-------|--------|
| **Alignment Library** | pairwise2 (deprecated) | pairwise2 (deprecated) |
| **Gap Open Penalty** | -3.0 | -3.0 |
| **Gap Extend Penalty** | -0.1 | -0.1 |
| **Length Threshold** | 0.3 | 0.3 |
| **Constraint Strength** | 0.4 × (1 - confidence) | 0.5 × (1 - confidence) |
| **Seq Distance Range** | 5.5 - 6.5 Å | 5.5 - 6.5 Å |
| **Target Distance** | 6.0 Å | 6.0 Å |
| **Random Scale Formula** | max(0.05, (0.5-sim)×0.15) | max(0.05, (0.5-sim)×0.25) |
| **Step Size Range** | 3.0 - 5.0 | 3.0 - 5.0 |
| **Comments** | Russian/English | English |

---

## Understanding the Approaches

### What is Template-Based Modeling (TBM)?

**TBM** is a traditional approach that:

1. **Finds similar sequences**: Searches training data for sequences similar to the query
2. **Aligns sequences**: Uses sequence alignment (BioPython's PairwiseAligner or pairwise2)
3. **Extracts templates**: Takes 3D coordinates from similar structures
4. **Adapts templates**: Maps template coordinates to query sequence positions
5. **Refines structures**: Applies geometric constraints and refinement

**TBM Code Pattern**:
```python
# Find similar sequences
similar_seqs = find_similar_sequences(query_seq, train_seqs, train_coords)
# Adapt template to query
adapted = adapt_template_to_query(query_seq, template_seq, template_coords)
# Apply constraints
refined = adaptive_rna_constraints(adapted, query_seq, confidence=similarity)
```

### What is RNAPro?

**RNAPro** is a deep learning model developed by NVIDIA for RNA 3D structure prediction, similar to AlphaFold but for RNA.

**RNAPro can operate in two modes**:

1. **Sequence-Only Mode** (De Novo Prediction)
   - Input: Just the RNA sequence
   - Uses MSA if available
   - Deep neural network + diffusion sampling
   - Output: Predicted 3D structure

2. **Template-Assisted Mode** (TBM Integration)
   - Input: RNA sequence + template structures
   - TBM finds templates
   - RNAPro's neural network refines/combines templates
   - Diffusion sampling generates final structure

**RNAPro Code Pattern**:
```python
# Model initialization
from rnapro.model.RNAPro import RNAPro
model = RNAPro(configs).to(device)
model.load_state_dict(checkpoint["model"])

# Inference
prediction, _, _ = model(
    input_feature_dict=data["input_feature_dict"],
    mode="inference",
)
```

### Hybrid TBM + RNAPro Approach

The hybrid approach (`rnapro-inference-with-tbm-87198f.ipynb`) combines both:

1. **Step 1**: Generate TBM predictions (fast, rule-based)
2. **Step 2**: Convert TBM to RNAPro template format
3. **Step 3**: RNAPro refines templates (slow, learned)
4. **Step 4**: Fallback to TBM for very long sequences

**Why Hybrid?**
- Better starting points: TBM provides good initial structures
- Handles edge cases: TBM more reliable for very similar templates
- Long sequences: RNAPro may struggle, TBM can handle them
- Ensemble diversity: Combining methods gives more diverse predictions

---

## Usage Recommendations

### For Template-Based Approaches:
- **Use #27JAN** as a baseline for comparison (moderate parameters)
- **Avoid 0.365** unless experimenting (has intentional parameter tuning, misleadingly named)
- **Note**: None of these use RNAPro despite 0.365's filename

### For Deep Learning Approaches:
- **Use `stanford-rna-3d-folding-pt2-rnapro-inference.ipynb`** for pure RNAPro
  - Best for sequences with good MSA coverage
  - Requires GPU
  - State-of-the-art accuracy

- **Use `rnapro-inference-with-tbm-87198f.ipynb`** for hybrid TBM + RNAPro
  - Best overall approach (combines TBM and RNAPro)
  - Handles edge cases better
  - Falls back to TBM for long sequences
  - Requires GPU

### For Production:
- **Use `part1-sub-1-4-4-hybrid-final-take-a0a437.ipynb`** for hybrid approach with multiple tools
  - Uses Boltz-1 + DRfold2 + AlphaFold3
  - Most computationally expensive
  - Handles edge cases best

---

## File Structure

```
RNA_stanford_notebooks/
├── stanford-rna-3d-folding-pt2-rnapro-inference.ipynb  # Pure RNAPro
├── rnapro-inference-with-tbm-87198f.ipynb             # Hybrid TBM + RNAPro
├── 0-365-rnapro-inference-with-tbm-tune.ipynb         # WARNING: TBM only (misleadingly named)
├── RNA FOLDING PREDICTION #27JAN.ipynb                # Baseline TBM
├── part1-sub-1-4-4-hybrid-final-take-a0a437.ipynb      # Hybrid (Boltz-1 + DRfold2 + AlphaFold3)
├── COMPARISON_ANALYSIS.md                              # Detailed comparison document
└── README.md                                           # This file
```

---

## Dependencies

### Common Dependencies
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `scipy` - Scientific computing
- `BioPython` - Sequence alignment and bioinformatics
  - `Bio.Align.PairwiseAligner` (modern API)
  - `Bio.pairwise2` (deprecated, used in older notebooks)
  - `Bio.Seq` - Sequence handling

### RNAPro Specific Dependencies
- `torch` - PyTorch for deep learning
- `rnapro` - RNAPro model library
- `biotite` - Structure I/O (CIF/PDB)
- GPU required for inference

### Part1 Hybrid Dependencies
- Boltz-1 model
- DRfold2 tool
- AlphaFold3 integration

---

## Competition Information

### Stanford RNA 3D Folding 2
- **Competition**: [Stanford RNA 3D Folding 2](https://www.kaggle.com/competitions/stanford-rna-3d-folding-2)
- **Data Path**: `/kaggle/input/stanford-rna-3d-folding-2/`
- **Data Files**:
  - `train_sequences.csv` - Training sequences
  - `test_sequences.csv` - Test sequences
  - `train_labels.csv` - Training structure coordinates
  - `validation_sequences.csv` - Validation sequences (if available)
  - `validation_labels.csv` - Validation structure coordinates (if available)

### Evaluation Metric
- TM-score (Template Modeling score)
- Compares predicted structures to ground truth
- Higher is better (range 0-1)

---

## Citations and Attribution

### Notebook Sources

- **RNAPro Notebooks**: Based on [RNAPro by NVIDIA](https://github.com/NVIDIA-Digital-Bio/RNAPro)
  - Model: [RNAPro-Private-Best-500M](https://huggingface.co/nvidia/RNAPro-Private-Best-500M)
  - Reference: [RNAPro Inference Notebook](https://www.kaggle.com/code/theoviel/stanford-rna-3d-folding-pt2-rnapro-inference) by theoviel

- **Part1 Hybrid**: 
  - Boltz-1: [Boltz-1 Inference Submission](https://www.kaggle.com/code/youhanlee/boltz-1-inference-submission) by youhanlee
  - DRfold2: Physics-based RNA structure refinement
  - AlphaFold3: DeepMind/Google DeepMind

- **TBM Notebooks**: Stanford RNA 3D Folding competition (Kaggle)

### External Tools and Libraries

- **BioPython**: Cock et al. (2009). "Biopython: freely available Python tools for computational molecular biology and bioinformatics." *Bioinformatics*, 25(11), 1422-1423.

---

## Additional Notes

1. **BioPython API Changes**: 
   - Modern notebooks should use `Bio.Align.PairwiseAligner`
   - Older notebooks (0.365, #27JAN) use deprecated `pairwise2` module

2. **Validation Data**: 
   - All notebooks attempt to combine train and validation data when available

3. **Language**: 
   - 0.365 contains Russian comments, suggesting it may have been developed by a different contributor

4. **Misleading Filename**: 
   - `0.365 RNAPro inference with TBM tune.ipynb` does NOT use RNAPro
   - Only contains TBM code despite filename suggesting RNAPro usage
   - For actual RNAPro, see the two notebooks listed in the RNAPro section

---

## Disclaimer

These notebooks are provided for educational and research purposes. If you use code or approaches from these notebooks, please:

1. Cite the original sources appropriately
2. Acknowledge the competition and data providers
3. Follow the competition's terms and conditions
4. Respect the licenses of external tools and libraries used

---

## Contact and Resources

- **Competition**: [Kaggle competition page](https://www.kaggle.com/competitions/stanford-rna-3d-folding-2)
- **RNAPro**: [GitHub repository](https://github.com/NVIDIA-Digital-Bio/RNAPro)
- **Boltz-1**: [Hugging Face](https://huggingface.co/boltz-community/boltz-1)

---

## Summary

This repository contains multiple approaches to RNA 3D structure prediction:

- **2 notebooks** use RNAPro (deep learning)
- **2 notebooks** use TBM only (template-based)
- **1 notebook** uses hybrid approach with other tools (Boltz-1, DRfold2, AlphaFold3)

**Key Takeaway**: The filename `0-365-rnapro-inference-with-tbm-tune.ipynb` is misleading - it does NOT use RNAPro, only TBM. For actual RNAPro implementations, use the two notebooks listed in the RNAPro section above.

# RNA Stanford Notebooks Comparison

This document compares the different RNA 3D folding prediction notebooks and their key differences.

## ⚠️ Important: RNAPro Usage Clarification

**Which notebooks actually use RNAPro?**

| Notebook | Uses RNAPro? | Method |
|----------|-------------|--------|
| `stanford-rna-3d-folding-pt2-rnapro-inference.ipynb` | ✅ **YES** | Pure RNAPro (deep learning) |
| `rnapro-inference-with-tbm-87198f.ipynb` | ✅ **YES** | Hybrid TBM + RNAPro |
| `0.365 RNAPro inference with TBM tune.ipynb` | ❌ **NO** | TBM only (misleadingly named) |
| `RNA FOLDING PREDICTION #27JAN.ipynb` | ❌ **NO** | TBM only |
| `0.358 RNA 3D Folding 2.ipynb` | ❌ **NO** | TBM only |
| `0.359 RNA FOLDING PREDICTION #25JAN.ipynb` | ❌ **NO** | TBM only |
| `part1-sub-1-4-4-hybrid-final-take-a0a437.ipynb` | ❌ **NO** | Boltz-1 + DRfold2 + AlphaFold3 |

**Key Finding**: Despite its filename, `0.365 RNAPro inference with TBM tune.ipynb` does NOT contain any RNAPro model code. It only uses Template-Based Modeling (TBM) with sequence alignment and geometric constraints. For actual RNAPro implementations, see the two notebooks listed above.

## Citations and Attribution

This repository contains notebooks developed for the Stanford RNA 3D Folding competition. Proper attribution is provided below:

### Notebook Sources

- **`0.358 RNA 3D Folding 2.ipynb`**: Template-based approach with optimized parameters
  - Source: Stanford RNA 3D Folding competition (Kaggle)
  - Competition: [Stanford RNA 3D Folding 2](https://www.kaggle.com/competitions/stanford-rna-3d-folding-2)
  
- **`0.359 RNA FOLDING PREDICTION #25JAN.ipynb`**: Baseline template-based approach
  - Source: Stanford RNA 3D Folding competition (Kaggle)
  - Competition: [Stanford RNA 3D Folding 2](https://www.kaggle.com/competitions/stanford-rna-3d-folding-2)
  
- **`0.365 RNAPro inference with TBM tune.ipynb`**: ⚠️ **Misleadingly named** - Template-based approach only (NO RNAPro)
  - Source: Stanford RNA 3D Folding competition (Kaggle)
  - Competition: [Stanford RNA 3D Folding 2](https://www.kaggle.com/competitions/stanford-rna-3d-folding-2)
  - **Note**: Despite the filename suggesting RNAPro usage, this notebook only uses Template-Based Modeling (TBM). No RNAPro model loading or inference code is present.
  
- **`RNA FOLDING PREDICTION #27JAN.ipynb`**: Template-based approach
  - Source: Stanford RNA 3D Folding competition (Kaggle)
  - Competition: [Stanford RNA 3D Folding 2](https://www.kaggle.com/competitions/stanford-rna-3d-folding-2)
  - Similar to 0.365 but with different parameter settings (constraint_strength=0.5, noise_scale=0.25)
  
- **`part1-sub-1-4-4-hybrid-final-take-a0a437.ipynb`**: Hybrid approach using Boltz-1, DRfold2, and AlphaFold3
  - Source: Stanford RNA 3D Folding competition (Kaggle)
  - Competition: [Stanford RNA 3D Folding](https://www.kaggle.com/competitions/stanford-rna-3d-folding)
  - **Boltz-1 Credit**: Based on [Boltz-1 Inference Submission](https://www.kaggle.com/code/youhanlee/boltz-1-inference-submission) by youhanlee
  - **DRfold2**: Uses DRfold2 for structure refinement
  - **AlphaFold3**: Integrates AlphaFold3 predictions for long sequences

- **`stanford-rna-3d-folding-pt2-rnapro-inference.ipynb`**: RNAPro deep learning model inference
  - Source: Stanford RNA 3D Folding competition (Kaggle)
  - Competition: [Stanford RNA 3D Folding 2](https://www.kaggle.com/competitions/stanford-rna-3d-folding-2)
  - **RNAPro Credit**: Based on [RNAPro by NVIDIA](https://github.com/NVIDIA-Digital-Bio/RNAPro)
  - Reference: [RNAPro Inference Notebook](https://www.kaggle.com/code/theoviel/stanford-rna-3d-folding-pt2-rnapro-inference) by theoviel

- **`rnapro-inference-with-tbm-87198f.ipynb`**: Hybrid RNAPro + Template-Based Modeling
  - Source: Stanford RNA 3D Folding competition (Kaggle)
  - Competition: [Stanford RNA 3D Folding 2](https://www.kaggle.com/competitions/stanford-rna-3d-folding-2)
  - **RNAPro Credit**: Based on [RNAPro by NVIDIA](https://github.com/NVIDIA-Digital-Bio/RNAPro)
  - **TBM Templates**: Uses template-based modeling to generate initial structures for RNAPro

### External Tools and Libraries

- **BioPython**: Used for sequence alignment (`Bio.Align.PairwiseAligner`, `pairwise2`)
  - Citation: Cock et al. (2009). "Biopython: freely available Python tools for computational molecular biology and bioinformatics." *Bioinformatics*, 25(11), 1422-1423.
  
- **Boltz-1**: Deep learning model for RNA structure prediction
  - Source: [Boltz-1 Community](https://huggingface.co/boltz-community/boltz-1)
  
- **DRfold2**: Physics-based RNA structure refinement tool
  - Used for energy minimization and structure optimization
  
- **AlphaFold3**: Protein and RNA structure prediction
  - Source: DeepMind/Google DeepMind

- **RNAPro**: Deep learning model for RNA structure prediction by NVIDIA
  - Source: [RNAPro GitHub](https://github.com/NVIDIA-Digital-Bio/RNAPro)
  - Model: [RNAPro-Private-Best-500M](https://huggingface.co/nvidia/RNAPro-Private-Best-500M)
  - Can work with or without templates (TBM)
  - Uses diffusion-based sampling for structure generation

### Competition Data

All notebooks use data from the Stanford RNA 3D Folding 2 competition:

- Competition: [Stanford RNA 3D Folding 2](https://www.kaggle.com/competitions/stanford-rna-3d-folding-2)
- Data path: `/kaggle/input/stanford-rna-3d-folding-2/`

### Disclaimer

These notebooks are provided for educational and research purposes. If you use code or approaches from these notebooks, please:
1. Cite the original sources appropriately
2. Acknowledge the competition and data providers
3. Follow the competition's terms and conditions
4. Respect the licenses of external tools and libraries used

## Notebook Overview

### 1. `0.358 RNA 3D Folding 2.ipynb`
**Approach**: Template-based with optimized parameters  
**Status**: Most refined template-based approach

### 2. `0.359 RNA FOLDING PREDICTION #25JAN.ipynb`
**Approach**: Template-based (simpler version)  
**Status**: Baseline template-based approach

### 3. `0.365 RNAPro inference with TBM tune.ipynb`
**Approach**: Template-based with tuned parameters  
**Status**: ⚠️ **Misleadingly named** - Does NOT use RNAPro, only TBM  
**Key Finding**: Despite the filename, this notebook contains no RNAPro model code. It only uses:
- Sequence alignment (BioPython pairwise2)
- Template adaptation
- Geometric constraints
- No deep learning components

### 3b. `RNA FOLDING PREDICTION #27JAN.ipynb`
**Approach**: Template-based  
**Status**: Baseline template-based approach  
**Differences from 0.365**: 
- Constraint strength: 0.5 (vs 0.4 in 0.365)
- Random noise scale: 0.25 (vs 0.15 in 0.365)

### 5. `stanford-rna-3d-folding-pt2-rnapro-inference.ipynb`
**Approach**: RNAPro deep learning model (can use templates optionally)  
**Status**: State-of-the-art deep learning approach

### 6. `rnapro-inference-with-tbm-87198f.ipynb`
**Approach**: Hybrid RNAPro + TBM (Template-Based Modeling)  
**Status**: Combines template-based modeling with RNAPro refinement

### 4. `part1-sub-1-4-4-hybrid-final-take-a0a437.ipynb`
**Approach**: Hybrid (Boltz-1 + DRfold2 + AlphaFold3)  
**Status**: Most advanced, production-ready hybrid approach

---

## Understanding RNAPro: Deep Learning vs Template-Based Modeling (TBM)

### What is RNAPro?

**RNAPro** is a deep learning model developed by NVIDIA for RNA 3D structure prediction. It's similar in concept to AlphaFold but specifically designed for RNA.

### How RNAPro Works

RNAPro can operate in **two modes**:

#### 1. **Sequence-Only Mode** (De Novo Prediction)
- **Input**: Just the RNA sequence
- **Process**: 
  - Uses Multiple Sequence Alignment (MSA) if available
  - Feeds sequence features into a deep neural network
  - Uses diffusion-based sampling to generate 3D coordinates
- **Output**: Predicted 3D structure

#### 2. **Template-Assisted Mode** (TBM Integration)
- **Input**: RNA sequence + template structures
- **Process**:
  - Finds similar sequences in training data (TBM step)
  - Uses their 3D structures as templates
  - RNAPro's neural network refines/combines templates
  - Diffusion sampling generates final structure
- **Output**: Template-guided predicted structure

### What is TBM (Template-Based Modeling)?

**TBM** stands for **Template-Based Modeling**. It's a traditional approach that:

1. **Finds similar sequences**: Searches training data for sequences similar to the query
2. **Aligns sequences**: Uses sequence alignment (e.g., BioPython's PairwiseAligner)
3. **Extracts templates**: Takes 3D coordinates from similar structures
4. **Adapts templates**: Maps template coordinates to query sequence positions
5. **Refines structures**: Applies geometric constraints and refinement

### RNAPro + TBM Hybrid Approach

The notebook `rnapro-inference-with-tbm-87198f.ipynb` shows a **hybrid approach**:

1. **Step 1 - Generate TBM predictions**: 
   - Uses traditional TBM to find templates and generate initial structures
   - Creates 5 template-based predictions

2. **Step 2 - Convert to RNAPro templates**:
   - Converts TBM predictions into template format for RNAPro
   - RNAPro can use these as starting points

3. **Step 3 - RNAPro refinement**:
   - RNAPro uses the TBM templates to guide its predictions
   - Neural network refines and improves the template-based structures
   - Generates 5 final predictions (one per template)

4. **Step 4 - Fallback strategy**:
   - For very long sequences (>1000 nucleotides), falls back to pure TBM
   - RNAPro may have memory/computational limits for very long sequences

### Key Differences: RNAPro vs Pure TBM

| Feature | Pure TBM (0.358, 0.359, 0.365, #27JAN) | RNAPro | RNAPro + TBM |
|---------|--------------------------------------|--------|--------------|
| **Method** | Sequence alignment + template adaptation | Deep learning (diffusion) | Hybrid (TBM → RNAPro) |
| **Input** | Sequence + training structures | Sequence (+ optional templates) | Sequence + TBM predictions |
| **Learning** | Rule-based heuristics | Learned from data | Learned + template guidance |
| **Accuracy** | Good for similar sequences | State-of-the-art | Best of both worlds |
| **Speed** | Fast | Slower (neural network) | Slowest (two-stage) |
| **Memory** | Low | High (GPU required) | Very high |
| **Notebooks** | 0.358, 0.359, 0.365, #27JAN | stanford-rna-3d-folding-pt2-rnapro-inference | rnapro-inference-with-tbm-87198f |

**⚠️ Important Note**: The notebook `0.365 RNAPro inference with TBM tune.ipynb` is **misleadingly named**. Despite suggesting RNAPro usage, it only contains TBM code. For actual RNAPro implementations, see:
- `stanford-rna-3d-folding-pt2-rnapro-inference.ipynb` (Pure RNAPro)
- `rnapro-inference-with-tbm-87198f.ipynb` (Hybrid TBM + RNAPro)

### Why Use TBM with RNAPro?

1. **Better starting points**: TBM provides good initial structures that RNAPro can refine
2. **Handles edge cases**: For sequences with very similar templates, TBM can be more reliable
3. **Long sequences**: RNAPro may struggle with very long sequences, TBM can handle them
4. **Ensemble diversity**: Combining TBM and RNAPro gives more diverse predictions

---

## Detailed Comparison

### Core Algorithm Differences

| Feature | 0.358 | 0.359 | 0.365 | #27JAN | part1 |
|---------|-------|-------|-------|--------|-------|
| **Alignment Library** | Bio.Align.PairwiseAligner | pairwise2 (deprecated) | pairwise2 (deprecated) | pairwise2 (deprecated) | Multiple (Boltz-1, DRfold2) |
| **Gap Open Penalty** | -8.0 | -8.0 | -3.0 | -3.0 | N/A (uses external tools) |
| **Gap Extend Penalty** | -0.3 | -0.3 | -0.1 | -0.1 | N/A |
| **Length Threshold** | 0.4 | 0.4 | 0.3 | 0.3 | N/A (handles all lengths) |
| **Constraint Strength** | 0.7 × (1 - confidence) | 0.5 × (1 - confidence) | 0.4 × (1 - confidence) | 0.5 × (1 - confidence) | Energy-based (DRfold2) |
| **Seq Distance Range** | 5.8 - 6.1 | 5.8 - 6.1 | 5.5 - 6.5 | 5.5 - 6.5 | N/A |
| **Target Distance** | 5.95 | 5.95 | 6.0 | 6.0 | N/A |
| **Random Scale Formula** | max(0.01, (0.4-sim)×0.1) | max(0.02, (0.5-sim)×0.1) | max(0.05, (0.5-sim)×0.15) | max(0.05, (0.5-sim)×0.25) | N/A |
| **Step Size Range** | 3.8 - 4.2 | 3.8 - 4.2 | 3.0 - 5.0 | 3.0 - 5.0 | N/A |
| **Uses RNAPro** | ❌ No | ❌ No | ❌ No (despite filename) | ❌ No | ❌ No (uses Boltz-1) |

### Key Code Differences

#### 1. Alignment Implementation

**0.358 (Modern API)**:
```python
from Bio.Align import PairwiseAligner
aligner = PairwiseAligner()
aligner.mode = 'global'
aligner.match_score = 2.0
aligner.mismatch_score = -1.0
aligner.open_gap_score = -8.0
aligner.extend_gap_score = -0.3
alignments = list(aligner.align(query_seq_obj, train_seq))
```

**0.359 & 0.365 (Deprecated API)**:
```python
from Bio import pairwise2
alignments = pairwise2.align.globalms(query_seq_obj, train_seq, 2, -1, -8, -0.3, one_alignment_only=True)
```

#### 2. Constraint Strength

- **0.358**: `constraint_strength = 0.7 * (1.0 - min(confidence, 0.95))`
  - Stronger constraints, more refinement for high-confidence templates
  
- **0.359**: `constraint_strength = 0.5 * (1.0 - min(confidence, 0.95))`
  - Moderate constraints
  
- **0.365**: `constraint_strength = 0.4 * (1.0 - min(confidence, 0.9))`
  - Weaker constraints, allows more flexibility

#### 3. Random Noise Addition

- **0.358**: `random_scale = max(0.01, (0.4 - similarity) * 0.1)`
  - Minimal noise for high-similarity templates
  
- **0.359**: `random_scale = max(0.02, (0.5 - similarity) * 0.1)`
  - Slightly more noise
  
- **0.365**: `random_scale = max(0.05, (0.5 - similarity) * 0.15)`
  - More noise, potentially for exploration

#### 4. Part1 Hybrid Approach

The `part1-sub-1-4-4-hybrid-final-take-a0a437.ipynb` uses a completely different strategy:

- **Boltz-1**: Deep learning model for initial predictions
- **DRfold2**: Physics-based refinement with energy minimization
- **AlphaFold3**: Integration for long sequences (>600 nucleotides)
- **Post-processing**: Combines predictions from different methods based on sequence length

Key features:
- Uses external tools (Boltz-1, DRfold2, AlphaFold3)
- Handles long sequences specially (uses Boltz-1 for sequences >600 nt)
- Converts mmCIF to PDB format
- Energy-based optimization instead of template matching

---

## Parameter Optimization Summary

### 0.358 (Most Optimized Template-Based)
- **Gap penalties**: -8.0/-0.3 (balanced for loop regions)
- **Distance constraints**: Narrow range (5.8-6.1) for consistency
- **Noise**: Minimal (0.01 minimum) for accuracy
- **Constraints**: Strong (0.7) for high-confidence templates

### 0.359 (Baseline)
- Standard template-based approach
- Moderate parameters
- Uses deprecated pairwise2

### 0.365 (Experimental) ⚠️ Misleadingly Named
- **Does NOT use RNAPro** despite filename
- Looser distance constraints (5.5-6.5)
- More noise for exploration (0.15 scale)
- Weaker constraints (0.4 strength)
- Different gap penalties (-3/-0.1)
- Contains Russian comments suggesting intentional "worsening" parameters

### #27JAN (Baseline TBM)
- Similar to 0.365 but with different parameters
- Constraint strength: 0.5 (vs 0.4 in 0.365)
- Random noise scale: 0.25 (vs 0.15 in 0.365)
- Uses same deprecated pairwise2 API

### Part1 (Hybrid)
- Combines multiple state-of-the-art methods
- Handles edge cases (long sequences, modified nucleotides)
- Most computationally intensive but potentially most accurate

---

## TM Score Metrics Comparison: Part1 vs Part2

### Current Status

**Part1**: The `part1-sub-1-4-4-hybrid-final-take-a0a437.ipynb` notebook is present in this repository.  
**Part2**: No part2 notebook was found in this repository. Part2 may refer to:
- A separate submission/notebook in a different location
- A different version of the competition (e.g., Stanford RNA 3D Folding Part 2)
- A future iteration of the approach

### TM Score Metrics Location

**Note**: TM (Template Modeling) score metrics are not present in the notebook files themselves. These metrics are typically:

1. **Computed during competition evaluation** on Kaggle/competition platforms
2. **Available in competition leaderboards** (public/private scores)
3. **Calculated using external evaluation scripts** that compare predicted structures to ground truth

### Where to Find TM Scores for Comparison

To compare TM scores for part1 and part2:

1. **Kaggle Competition Leaderboard**:
   - Check the Stanford RNA 3D Folding competition page
   - Look for submissions named "part1" and "part2" (or similar naming)
   - Compare public and private scores
   - Note: Leaderboard scores may use different metrics (e.g., GDT-TS, RMSD, TM-score)

2. **Submission Metadata**:
   - Check if there are any evaluation result files in the repository
   - Look for CSV files with evaluation metrics
   - Check Kaggle notebook outputs for evaluation results

3. **External Evaluation**:
   - Use tools like `TMscore` or `US-align` to compute TM scores locally
   - Compare predicted structures against ground truth PDB files
   - Run evaluation scripts on both part1 and part2 submission files

### Expected TM Score Range

For RNA structure prediction:
- **TM Score > 0.5**: Generally considered a good prediction
- **TM Score > 0.7**: High-quality prediction
- **TM Score < 0.3**: Poor prediction

### Part1 Approach Summary

The part1 notebook (`part1-sub-1-4-4-hybrid-final-take-a0a437.ipynb`) uses:
- **Hybrid approach**: Boltz-1 + DRfold2 + AlphaFold3
- **Special handling**: Long sequences (>600 nucleotides) use Boltz-1
- **Post-processing**: Combines predictions from different methods
- **Expected performance**: Likely higher than template-based approaches due to:
  - Use of state-of-the-art deep learning models (Boltz-1)
  - Physics-based refinement (DRfold2)
  - Integration with AlphaFold3 for long sequences

### How to Compare Part1 vs Part2 (When Available)

1. **If Part2 notebook is found elsewhere**:
   - Add it to this repository
   - Run both notebooks on the same test set
   - Compare outputs using evaluation scripts

2. **If Part2 refers to competition submissions**:
   - Check competition leaderboard for both submissions
   - Compare public/private scores
   - Analyze which approach works better for different sequence types

3. **Manual Evaluation**:
   ```bash
   # Example: Compare structures using TMscore
   # (Assuming you have predicted PDB files and ground truth)
   TMscore part1_prediction.pdb ground_truth.pdb
   TMscore part2_prediction.pdb ground_truth.pdb
   ```

---

## Recommendations

### For Template-Based Approaches:
- **Use 0.358** for best template-based results (most optimized parameters, modern API)
- **Use #27JAN** as a baseline for comparison (moderate parameters)
- **Avoid 0.365** unless experimenting (has intentional "worsening" parameters based on comments, misleadingly named)
- **Note**: None of these use RNAPro despite 0.365's filename suggesting otherwise

### For Production:
- **Use part1** for best overall results (hybrid approach with multiple methods: Boltz-1 + DRfold2 + AlphaFold3)
- **Use RNAPro notebooks** for state-of-the-art deep learning predictions:
  - `stanford-rna-3d-folding-pt2-rnapro-inference.ipynb` for pure RNAPro
  - `rnapro-inference-with-tbm-87198f.ipynb` for hybrid TBM + RNAPro (best of both worlds)
- More computationally expensive but handles edge cases better

### For Development:
- **Use 0.359** as a baseline for comparison
- Simpler codebase, easier to understand and modify

---

## File Structure

```
RNA_stanford_notebooks/
├── 0.358 RNA 3D Folding 2.ipynb                    # Optimized template-based
├── 0.359 RNA FOLDING PREDICTION #25JAN.ipynb      # Baseline template-based
├── 0.365 RNAPro inference with TBM tune.ipynb      # ⚠️ TBM only (misleadingly named)
├── RNA FOLDING PREDICTION #27JAN.ipynb             # Baseline template-based
├── stanford-rna-3d-folding-pt2-rnapro-inference.ipynb  # Pure RNAPro (deep learning)
├── rnapro-inference-with-tbm-87198f.ipynb         # Hybrid TBM + RNAPro
├── part1-sub-1-4-4-hybrid-final-take-a0a437.ipynb  # Hybrid (Boltz-1 + DRfold2 + AlphaFold3)
└── README.md                                        # This file
```

---

## Additional Notes

1. **BioPython API Changes**: 0.358 uses the newer `Bio.Align.PairwiseAligner` API, while 0.359, 0.365, and #27JAN use the deprecated `pairwise2` module.

2. **Validation Data**: All notebooks attempt to combine train and validation data when available.

3. **Part1 Complexity**: The part1 notebook is significantly more complex, integrating multiple external tools and requiring GPU resources.

4. **Language**: 0.365 contains Russian comments, suggesting it may have been developed by a different contributor.

5. **⚠️ Misleading Filename**: The notebook `0.365 RNAPro inference with TBM tune.ipynb` does NOT actually use RNAPro. Despite the filename suggesting RNAPro usage, it only contains Template-Based Modeling code. For actual RNAPro implementations, see:
   - `stanford-rna-3d-folding-pt2-rnapro-inference.ipynb` (Pure RNAPro)
   - `rnapro-inference-with-tbm-87198f.ipynb` (Hybrid TBM + RNAPro)

6. **RNAPro vs TBM**: Only two notebooks actually use RNAPro:
   - `stanford-rna-3d-folding-pt2-rnapro-inference.ipynb`: Pure RNAPro with optional templates
   - `rnapro-inference-with-tbm-87198f.ipynb`: Hybrid approach (TBM generates templates → RNAPro refines)

---

## Future Work

To properly compare TM scores:
1. Run evaluation scripts on all notebook outputs
2. Compare against ground truth structures
3. Document results in this README
4. Identify which approach works best for different sequence types

---

## Usage and Attribution Guidelines

### When Using These Notebooks

If you use code, approaches, or ideas from these notebooks:

1. **Cite the Competition**: Acknowledge the Stanford RNA 3D Folding 2 competition
2. **Cite External Tools**: Properly cite Boltz-1, DRfold2, AlphaFold3, and BioPython
3. **Respect Licenses**: Check and comply with licenses of all external tools and libraries
4. **Acknowledge Sources**: If you adapt code from specific Kaggle notebooks, cite them appropriately

### Recommended Citation Format

If you use these notebooks in your research or work, consider citing:

```
Stanford RNA 3D Folding 2 Competition Notebooks
Competition: https://www.kaggle.com/competitions/stanford-rna-3d-folding-2
Boltz-1 Reference: https://www.kaggle.com/code/youhanlee/boltz-1-inference-submission
```

### License and Terms

- These notebooks are provided as-is for educational and research purposes
- Users are responsible for ensuring compliance with:
  - Competition terms and conditions
  - Licenses of external tools (Boltz-1, DRfold2, AlphaFold3, etc.)
  - Data usage agreements
  - Any applicable academic or commercial use restrictions

### Contact

For questions about:
- **Competition**: See the [Kaggle competition page](https://www.kaggle.com/competitions/stanford-rna-3d-folding-2)
- **External Tools**: Refer to the respective tool documentation and licensing
- **This Repository**: This is a comparison document; individual notebooks may have their own authors/contributors

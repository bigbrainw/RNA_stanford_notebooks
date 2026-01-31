# RNA Folding Prediction - January 30th Version Documentation

## Overview

This notebook implements a **simplified and optimized** RNA 3D structure prediction pipeline. Compared to the January 27th version, this version focuses on **speed and simplicity** while maintaining competitive accuracy.

## Key Features

### 1. **Sequence Cleaning**
- **Nucleotide Mapping**: Handles modified nucleotides and non-standard bases
- Maps uncommon nucleotides (I, 1MA, PSU, M2G, 5MC, T) to standard bases
- Ensures consistent sequence representation for alignment

```python
NUCLEOTIDE_MAPPING = {
    'A': 'A', 'U': 'U', 'G': 'G', 'C': 'C',
    'I': 'A', '1MA': 'A', 'PSU': 'U', 'M2G': 'G', '5MC': 'C', 'T': 'U',
}
```

### 2. **Simplified Constraint Refinement**
- **Two-pass refinement** with distance constraints
- **Adjacent residue constraints**: Maintains ~5.95Å distance between consecutive residues
- **Skip-residue constraints**: Maintains ~10.2Å distance between residues i and i+2
- **Adaptive strength**: Constraint strength decreases with template confidence

**Key Parameters:**
- Target distance: 5.95Å (adjacent), 10.2Å (skip)
- Strength multiplier: 0.68 × (1 - confidence)
- Adjustment factors: 0.45 (adjacent), 0.25 (skip)

### 3. **Template Adaptation**
- Uses **pairwise2** for sequence alignment (simpler than modern aligner)
- **Gap filling**: Linear interpolation for gaps between aligned regions
- **Edge cases**: Handles gaps at sequence ends with fixed 3.5Å extension

**Alignment Parameters:**
- Match score: 2.0
- Mismatch score: -1.0
- Gap open: -7.0
- Gap extend: -0.25

### 4. **Similar Sequence Finding**
- **Direct pairwise alignment** approach (no k-mer pre-filtering)
- **Length filtering**: Only considers sequences within 40% length difference
- **Score normalization**: Normalizes alignment score by sequence length
- Returns top 5 most similar sequences

### 5. **Structure Prediction**
- **Template-based prediction**: Uses top similar sequences as templates
- **Noise strategy**: 
  - Slot 0: **No noise** (cleanest prediction)
  - Slots 1-4: **Minimal noise** based on template confidence
- **Fallback**: Simple linear chain for missing predictions

**Noise Formula:**
```python
noise = 0.0 if i == 0 else max(0.006, (0.38 - sim) * 0.07)
```

## Architecture Comparison: Jan 27 vs Jan 30

| Feature | Jan 27 Version | Jan 30 Version |
|---------|---------------|----------------|
| **Pre-filtering** | K-mer similarity (fast) | Direct alignment |
| **Alignment** | Modern PairwiseAligner + pairwise2 fallback | pairwise2 only |
| **Constraints** | Multiple iterations, steric clash prevention | Two-pass, simpler constraints |
| **Memory Management** | Incremental writing, GC, monitoring | Accumulates in memory |
| **Noise Strategy** | All slots get noise | Slot 0 clean, others minimal |
| **Error Handling** | Try-except with fallbacks | Direct execution |
| **Code Complexity** | ~500 lines | ~170 lines |

## Performance Characteristics

### Advantages of Jan 30 Version:
1. **Simpler codebase**: Easier to understand and debug
2. **Faster execution**: No k-mer overhead, simpler constraints
3. **Clean first prediction**: Slot 0 has no noise for best accuracy
4. **Lower memory overhead**: No monitoring/GC overhead

### Trade-offs:
1. **Slower for large datasets**: No k-mer pre-filtering means more alignments
2. **Memory accumulation**: Stores all predictions before writing (may fail on large datasets)
3. **Less robust**: No error handling or fallbacks
4. **Simpler constraints**: May be less accurate for complex structures

## Code Structure

```
Phase 1: Data Loading
  ├── Load train/test sequences and labels
  ├── Process labels into coordinate dictionary
  └── Clean sequences using nucleotide mapping

Phase 2: Constraint Refinement
  └── adaptive_rna_constraints() - Two-pass distance refinement

Phase 3: Template Adaptation
  └── adapt_template_to_query() - Align and map coordinates

Phase 4: Similarity Search & Prediction
  ├── find_similar_sequences() - Find top templates
  └── predict_rna_structures() - Generate 5 predictions

Phase 5: Output Generation
  └── Process all test sequences and write submission.csv
```

## Key Design Decisions

### 1. **Why No K-mer Pre-filtering?**
- Simpler implementation
- For small-medium datasets, direct alignment is acceptable
- Reduces code complexity

### 2. **Why Clean Slot 0?**
- Ensures at least one high-quality prediction
- Reduces noise in best-case scenario
- Common practice in ensemble predictions

### 3. **Why Simpler Constraints?**
- Faster execution
- Two-pass refinement is sufficient for most cases
- Avoids memory-intensive steric clash detection

### 4. **Why Accumulate in Memory?**
- Simpler code
- For 28 test sequences, memory is manageable
- Faster I/O (single write vs incremental)

## Usage Notes

### When to Use Jan 30 Version:
- ✅ Small to medium datasets (<1000 sequences)
- ✅ When speed is more important than robustness
- ✅ When you want simpler, more maintainable code
- ✅ When memory is not a concern

### When to Use Jan 27 Version:
- ✅ Large datasets (>1000 sequences)
- ✅ When robustness and error handling are critical
- ✅ When memory constraints are tight
- ✅ When you need detailed progress monitoring

## Potential Improvements

1. **Add incremental writing** for memory efficiency
2. **Add k-mer pre-filtering** for large datasets
3. **Add error handling** for robustness
4. **Add memory monitoring** for large-scale runs
5. **Optimize alignment** with early termination

## Performance Metrics

- **Code size**: ~170 lines (vs ~500 in Jan 27)
- **Execution time**: Faster due to simpler operations
- **Memory usage**: Lower overhead, but accumulates results
- **Accuracy**: Comparable (0.365 score maintained)

## Conclusion

The January 30th version represents a **streamlined approach** that prioritizes simplicity and speed. It's ideal for scenarios where the dataset is manageable and you want clean, maintainable code. For production environments with large datasets or strict memory constraints, the January 27th version with its optimizations may be more suitable.

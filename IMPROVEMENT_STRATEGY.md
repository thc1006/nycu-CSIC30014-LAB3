# Improvement Strategy to Reach 91%+ Accuracy

## Current Status
- **Best Score**: 83.90% (submission_improved.csv)
- **Target**: 91.085% (first place)
- **Gap**: +7.185% needed

## Ultra-Deep Analysis Key Findings

### 1. CRITICAL: Extreme Class Imbalance (47.2:1 ratio)
```
Training Set (3234 samples):
- bacteria:  1512 (46.75%)
- normal:     863 (26.69%)
- virus:      827 (25.57%)
- COVID-19:    32 (0.99%)  ⚠️ Only 32 samples!
```

**Impact**: Model struggles to learn COVID-19 features
- Each K-Fold validation has only 6-7 COVID-19 samples
- Extremely difficult for the model to generalize

**Solution Applied**:
- Focal Loss alpha increased from 12.0 → 20.0 for COVID-19
- Focal Loss gamma increased from 2.5 → 4.0 (more focus on hard examples)
- Label smoothing increased from 0.12 → 0.15

### 2. WARNING: Overfitting (Avg Confidence 0.990)
**Finding**: Most single-fold models show perfect confidence (1.0)
- This indicates severe overfitting to training data
- Ensemble has more reasonable confidence (0.91)

**Solution Applied**:
- Dropout increased from 0.25 → 0.35 (ConvNeXt) / 0.40 (EfficientNet-V2-L)
- More aggressive data augmentation:
  - Rotation: 10° → 15°
  - Random erasing: 0.2 → 0.3
  - Mixup prob: 0.6 → 0.7
  - CutMix prob: 0.4 → 0.5
  - Added vertical flip and Gaussian noise
- Weight decay increased from 0.00015 → 0.00025

### 3. Image Statistics Analysis
```
Average Dimensions: 1321x964 ±389 pixels
Current Usage: 384px (only 29% of original resolution!)
```

**Solution Applied**:
- ConvNeXt-Base: 448px input (39% larger than current)
- EfficientNet-V2-L: 480px input (53% larger than current)
- Progressive resizing: Start at 384px, gradually increase to full resolution

### 4. Model Diversity in Ensemble
**Finding**: Different fold models make very different predictions
- Fold 1 predicts 669 bacteria vs Fold 0 predicts 542 bacteria
- This high variance suggests ensemble could benefit from diversity

**Solution Applied**:
- Train multiple architecture types:
  - ConvNeXt-Base (transformer-based)
  - EfficientNet-V2-L (CNN-based)
  - Different from current EfficientNet-V2-S
- Use TTA (Test Time Augmentation) for each model

## Implementation Plan

### Phase 1: Train Ultra-Optimized Models (Current)

#### Model 1: ConvNeXt-Base
```bash
python3 -m src.train_v2 --config configs/ultra_optimized.yaml
```
**Key Features**:
- Larger capacity (88M parameters vs 21M in EfficientNet-V2-S)
- 448px input size
- Focal Loss alpha=20.0 for COVID-19
- Stochastic Weight Averaging (SWA)
- Model EMA (Exponential Moving Average)
- Progressive resizing

**Expected Improvement**: +2-3% (85-87%)

#### Model 2: EfficientNet-V2-L
```bash
python3 -m src.train_v2 --config configs/efficientnet_v2_l.yaml
```
**Key Features**:
- Very large model (118M parameters)
- 480px input size
- Same extreme class weighting
- Higher dropout (0.4) and regularization

**Expected Improvement**: +2-3% (85-87%)

### Phase 2: Test Time Augmentation (TTA)

Apply TTA to all models using `src/predict_tta.py`:
```bash
python3 -m src.predict_tta \
  --checkpoints outputs/ultra_optimized/best.pt outputs/efficientnet_v2_l/best.pt \
  --test-csv data/test_data.csv \
  --test-images test_images \
  --output data/submission_tta_ultra.csv \
  --tta-transforms 5
```

**Expected Improvement**: +1-2% (TTA typically adds 1-2%)

### Phase 3: Advanced Ensemble

Combine multiple sources:
1. ConvNeXt-Base + TTA
2. EfficientNet-V2-L + TTA
3. Existing best model (submission_improved.csv) + TTA

```bash
python3 ensemble_probabilities.py \
  --submissions data/submission_convnext_tta.csv \
                data/submission_effv2l_tta.csv \
                data/submission_improved_tta.csv \
  --output data/submission_final_ensemble.csv \
  --method geometric_mean
```

**Expected Improvement**: +1-2% (ensemble diversity)

### Phase 4: (Optional) Pseudo-Labeling

If still below 91%:
1. Use ensemble to pseudo-label test set
2. Retrain with high-confidence test predictions
3. Apply only to majority classes (avoid COVID-19 due to scarcity)

## Expected Results

| Phase | Method | Expected Accuracy | Cumulative |
|-------|--------|-------------------|------------|
| Baseline | Current Best | 83.90% | 83.90% |
| Phase 1 | Larger Models | +2.5% | 86.40% |
| Phase 2 | TTA | +1.5% | 87.90% |
| Phase 3 | Advanced Ensemble | +2.0% | 89.90% |
| Phase 4 | Pseudo-Labeling | +1.5% | **91.40%** ✓ |

## Risk Mitigation

### Risk 1: Insufficient Training Data for COVID-19
**Mitigation**:
- Heavy augmentation on COVID-19 samples
- Class-specific augmentation (can create 10x more COVID-19 samples)
- Transfer learning from large medical imaging datasets

### Risk 2: Overfitting to Validation Set
**Mitigation**:
- K-Fold Cross Validation (already implemented)
- Strong regularization
- Monitor train-val gap closely

### Risk 3: GPU Memory Limitations
**Models too large for single GPU**:
- Use gradient accumulation (simulate larger batch size)
- Mixed precision training (FP16)
- Reduce batch size if needed

## Monitoring Plan

During training, monitor:
1. **Train-Val Gap**: Should be < 5% (overfitting check)
2. **Per-Class Accuracy**: COVID-19 should be > 50% (class imbalance check)
3. **Confidence Distribution**: Should be more spread out than 0.990 avg
4. **GPU Utilization**: Should be > 90% (efficiency check)

## Timeline

- **Phase 1** (Train 2 models): ~4-6 hours (parallel on 1 GPU)
- **Phase 2** (TTA inference): ~30 minutes per model
- **Phase 3** (Ensemble): ~5 minutes
- **Phase 4** (If needed): ~3-4 hours

**Total**: 5-11 hours to reach 91%+ target

## Success Criteria

✓ Reach 91.085%+ on Kaggle public leaderboard
✓ Per-class accuracy all > 80%
✓ COVID-19 F1-score > 0.7 (currently challenging)
✓ Ensemble diversity (models disagree on ~20-30% of samples)

## Next Steps

1. ✓ Complete ultra-deep analysis
2. ✓ Create ultra-optimized configurations
3. Start training ConvNeXt-Base model
4. Start training EfficientNet-V2-L model
5. Apply TTA to all models
6. Create final ensemble
7. Submit to Kaggle and iterate based on results

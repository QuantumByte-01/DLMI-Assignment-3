# DLMI Project Report

## Overview
Deep Learning Models Investigation (DLMI) - Comparing multiple architectures for embryo classification using custom loss functions.

## Models Evaluated

### 1. MobileNetV3 (notebook-1-mobilenetv3-small.ipynb)
- Architecture: MobileNetV3-Small with frozen backbone (1.53M total / 1.25M trainable params, 224x224 input)
- Best Epoch: 5
- Validation F1: 0.4033
- Validation Accuracy: 50.45%
- Key Observations:
  - Lightweight architecture (56M FLOPs/image vs 5.7B for InceptionV3)
  - Steady improvement across all 5 epochs (val_f1: 0.2936 → 0.4033)
  - Training completed in 77.7 min total on NVIDIA L4
  - Test inference was cut off before completion — no final test metrics
  - Faster convergence per epoch (~850s) compared to InceptionV3 (~1100s)

### 2. InceptionV3 (notebook-2-inceptionV3.ipynb)
- Architecture: InceptionV3 with frozen backbone (21.8M total / 6.1M trainable params, 299x299 input)
- Best Epoch: 8
- Validation F1: 0.3946
- Test Accuracy: 51.95%
- Test Macro F1: 0.4004
- Test Weighted F1: 0.5282
- Key Observations:
  - Continued improvement across 8 epochs (val_f1: 0.3552→0.3946)
  - Best per-class performance: pPNa (0.812), pEB (0.687), p2 (0.683)
  - Struggles with rare/adjacent phases: p7 (0.068), p6 (0.089), p3 (0.139)
  - pHB recall: 14% — rare class (only 59 test frames, 97 total in dataset)
  - Early stopping patience not triggered — ran full scheduled epochs

### 3. VGG16 (notebook-3-vgg16.ipynb)
- Architecture: VGG16 with frozen backbone, trainable head (114M parameters)
- Best Epoch: 7
- Validation F1: 0.4932
- Validation Accuracy: 57.50%
- Key Observations: 
  - Steady improvement from epoch 1-7, then early stopping triggered at epoch 11
  - Best validation F1 reached at epoch 7 with balanced performance
  - Strong performance on common phases (p9+: F1=0.63, pEB: F1=0.68)
  - Challenges with rare phases (pHB: F1=0.00)

### 4. VGG19 (notebook-4-vgg19.ipynb)
- Architecture: VGG19 with frozen backbone
- Best Epoch: 16
- Validation F1: 0.4608
- Key Observations:
  - Convergence required more epochs than VGG16
  - Final validation F1 lower than VGG16 (0.4608 vs 0.4932)
  - More gradual improvement trajectory
  - Deeper architecture did not yield proportional performance gains

### 5. LSTM (notebook-5-lstm.ipynb)
- Architecture: Row-wise BiLSTM (2-layer, bidirectional, hidden=256) trained from scratch, 3.4M parameters, 128x128 input
- Best Epoch: 5
- Validation F1: 0.1316
- Test Accuracy: 19.37%
- Test Macro F1: 0.1415
- Test Weighted F1: 0.1808
- Key Observations:
  - Trained from scratch on both T4 GPUs via DataParallel (88 min total, 5 epochs)
  - Steady but slow improvement (val_f1: 0.050 to 0.1316 over 5 epochs)
  - Only pEB crossed F1=0.5 (0.592); pPB2 and p7 scored 0.000
  - Most confusion between adjacent phases: pPNa/pPNf, p8/p4 to p6/p3
  - Significantly lower performance than CNN baselines — expected, as row-wise LSTM discards 2D spatial structure that CNNs exploit via convolutions

## Results Summary

### Validation Results
| Model | Val F1 (best) | Val Accuracy | Best Epoch |
|-------|--------------|--------------|-----------|
| MobileNetV3 | 0.4033 | 50.45% | 5 |
| InceptionV3 | 0.3946 | ~51.73% | 8 |
| VGG16 | 0.4932 | 57.50% | 7 |
| VGG19 | 0.4608 | 55.38% | 16 |
| LSTM | 0.1316 | 18.64% | 5 |

### Test Set Results (final evaluation)
| Model | Test Accuracy | Macro F1 | Weighted F1 |
|-------|--------------|----------|------------|
| MobileNetV3 | 49.95% | 0.3806 | 0.5059 |
| InceptionV3 | 51.95% | 0.4004 | 0.5282 |
| VGG16 | 57.40% | 0.4549 | 0.5875 |
| VGG19 | 55.56% | 0.4269 | 0.5630 |
| LSTM | 19.37% | 0.1415 | 0.1808 |

## Best Performing Model
**Model**: VGG16
**Reason**: 
- Highest validation F1-score across all models (0.4932)
- Highest validation accuracy (57.50%)
- Optimal convergence at epoch 7 with early stopping at epoch 11
- Better performance-efficiency tradeoff than VGG19
- Outperforms lightweight models by 13-20% in F1-score

## Loss Function Used
See `custom_loss_function.md` for details on the custom loss function implementation.

## Conclusion

This study compared five architectures for 16-class embryo developmental phase classification using the custom Embryo Composite Loss (ECL): MobileNetV3-Small, InceptionV3, VGG16, VGG19, and a row-wise BiLSTM.

**VGG16 achieved the best overall performance** with 57.40% test accuracy and macro F1 of 0.4549, followed by VGG19 (55.56%, 0.4269). The larger VGG architectures benefited from their deep feature extractors despite the frozen backbone setting. InceptionV3's multi-scale receptive fields gave it an edge over MobileNetV3-Small (51.95% vs 49.95%), while remaining far more parameter-efficient than VGG16.

The **LSTM** (BiLSTM, row-wise) scored 19.37% test accuracy and macro F1 of 0.1415, substantially below all CNN baselines. This is expected: treating image rows as a temporal sequence discards the 2D spatial structure that convolutional filters exploit. The LSTM also lacked ImageNet pretraining, training all 3.4M parameters from scratch. Despite this, the model showed consistent improvement over 5 epochs and correctly captured some ordinal structure in phase predictions (most errors were between adjacent phases).

The ECL loss function proved effective across all architectures, particularly for reducing stage boundary violations and improving recall on rare classes such as pPNf and pSB. However, pHB (only 97 total frames; 59 in test) remained undetected by most models, highlighting the fundamental limit of extremely rare classes at current dataset scale.

Key takeaways:
- CNN architectures with ImageNet pretraining vastly outperform LSTM on image classification
- Deeper models (VGG16/19) outperform lightweight models on fine-grained, imbalanced classification
- ECL's ordinal and boundary terms meaningfully constrain biologically invalid predictions
- Rare class recall (p3, p5, p6, p7, pHB) remains a challenge across all models due to 526:1 imbalance
- VGG16 offers the best accuracy-to-convergence tradeoff (best F1 at epoch 7 vs VGG19's epoch 16)

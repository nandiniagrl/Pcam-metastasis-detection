# PCam Metastasis Detection — Full Report

> **Status of this document.** This report has been regenerated from the uploaded `results/` artifact. Numerical placeholders from the original template have been replaced using the metrics, histories, ablation reports, robustness summaries, and figures available in the results files.

## 1. Problem statement

We address binary classification of 96×96 H&E-stained histopathology patches into **normal** vs **tumour-containing** tissue. The dataset is **PatchCamelyon (PCam)**, a patch-level benchmark derived from the CAMELYON16 whole-slide-image challenge. The task is clinically motivated because lymph-node metastasis detection is important for cancer staging and treatment planning.

A reliable automated metastasis detector has direct clinical value as a **decision-support component**. In a practical workflow, a patch-level classifier can be used to score small tissue regions and help prioritize suspicious areas for expert review. The model is not intended to replace a pathologist; it is intended to support screening, triage, and review consistency.

## 2. Dataset

PCam contains 327,680 colour histopathology patches of size 96×96 pixels. It uses predefined train/validation/test splits and binary labels.

| Property | Value |
|---|---|
| Dataset | PatchCamelyon (PCam) |
| Source | CAMELYON16 lymph-node histopathology slides |
| Image type | H&E-stained RGB tissue patches |
| Image size | 96 × 96 pixels |
| Task | Binary classification |
| Classes | Normal / Tumour |
| Train split | 262,144 patches |
| Validation split | 32,768 patches |
| Test split | 32,768 patches |

A positive label indicates tumour tissue in the central region of the patch. This matters for preprocessing: aggressive cropping or large spatial shifts can change the label semantics, so transformations must be chosen carefully.

![PCam sample patches](../results/figures/sample_patches.png)

## 3. Goal of the project

The goal was to build a reproducible medical image classification pipeline for PCam metastasis detection, compare multiple modelling approaches, and interpret the results from both technical and clinical perspectives.

The project specifically focuses on:

- data loading, preprocessing, and augmentation for histopathology patches;
- baseline comparison between classical machine learning and deep learning;
- transfer learning with ResNet50;
- metric-based evaluation using clinically relevant measures;
- robustness testing under rotations and flips;
- interpretability using Grad-CAM;
- limitations, bias, and ethical considerations for clinical use.

## 4. Evaluation protocol

Evaluation was performed on the official PCam test set. The following metrics were used:

- **Accuracy** — overall correctness; useful because PCam is balanced.
- **Precision** — of patches predicted as tumour, how many were truly tumour.
- **Recall** — of all tumour patches, how many were detected; clinically important because false negatives can miss cancer.
- **F1-score** — balance between precision and recall.
- **ROC-AUC** — threshold-independent ranking quality.
- **NLL** — probability quality / calibration-sensitive loss.
- **Confusion matrix** — direct inspection of true/false positive and negative counts.

## 5. Methods and modelling approaches

Three model implementations were evaluated:

| Model | Role | Modelling approach | Interpretation |
|---|---|---|---|
| Logistic Regression | Classical baseline | Classical ML using image features | Tests whether simple linear decision boundaries are sufficient. |
| Custom CNN | Deep learning baseline | CNN trained from scratch | Learns local tissue texture features directly from PCam. |
| ResNet50 | Final transfer-learning model | ImageNet-pretrained CNN fine-tuned for PCam | Tests whether pretrained visual features improve discrimination and interpretability. |

This satisfies the requirement for **two baselines** and **multiple modelling approaches**. Logistic Regression is the classical baseline, Custom CNN is the deep learning baseline, and ResNet50 is the final transfer-learning model.

## 6. Preprocessing and augmentation

The pipeline loads PCam HDF5 files and applies PyTorch-style transformations dynamically during training. Images are converted to tensors and fed through a DataLoader into the selected model.

Training-time augmentation was used because histopathology tissue does not have a fixed natural orientation. The augmentation strategy included:

- horizontal flips;
- vertical flips;
- rotations;
- tensor conversion / scaling;
- mild colour and geometric transformations in the ablation pipeline.

Aggressive cropping was avoided because PCam labels depend on the centre region of each patch. This is important: cropping or strong translation may remove or move the labelled tumour evidence.

## 7. Training setup and reproducibility

The result artifact contains saved histories, test metrics, ablation reports, figures, and robustness summaries. Neural model histories were saved for seed 42, while ablation studies were run across seeds 42, 123, and 2024.

| Model | Epochs run | Best validation loss | Best validation accuracy | Best validation recall | Best validation F1 | Best validation ROC-AUC | Final train loss | Final validation loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Custom CNN | 10 | 0.281 | 0.887 | 0.896 | 0.888 | 0.956 | 0.141 | 0.305 |
| ResNet50 | 4 | 0.307 | 0.873 | 0.805 | 0.864 | 0.954 | 0.121 | 0.380 |


Interpretation:

- The Custom CNN trained for 10 epochs and achieved its best validation ROC-AUC of **0.956**.
- ResNet50 trained for 4 epochs in the saved history and achieved its best validation ROC-AUC of **0.954**.
- ResNet50 reached strong validation performance quickly, consistent with transfer learning.
- Custom CNN continued to improve over a longer training history and reached a strong validation F1-score.

## 8. Main test results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | NLL | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.615 | 0.613 | 0.622 | 0.617 | 0.655 | 0.709 | 9950 | 6441 | 6187 | 10190 |
| Custom CNN | 0.847 | 0.885 | 0.797 | 0.838 | 0.921 | 0.402 | 14689 | 1702 | 3327 | 13050 |
| ResNet50 | 0.827 | 0.936 | 0.702 | 0.802 | 0.929 | 0.428 | 15604 | 787 | 4888 | 11489 |


![Model comparison](../results/figures/model_comparison.png)

### Main interpretation

- Logistic Regression performed worst, with **ROC-AUC = 0.655** and **accuracy = 0.615**. This shows that simple linear modelling is not sufficient for complex histopathology texture patterns.
- Custom CNN achieved the highest **test accuracy (0.847)**, **recall (0.797)**, and **F1-score (0.838)**. This makes it the best balanced classifier at the default 0.5 threshold.
- ResNet50 achieved the highest **precision (0.936)** and **ROC-AUC (0.929)**. This indicates stronger ranking ability and more conservative tumour predictions.
- Compared with Logistic Regression, Custom CNN improved accuracy by **0.232 absolute points** and ROC-AUC by **0.265**. ResNet50 improved ROC-AUC by **0.274** over Logistic Regression.

The results show a clear benefit from deep learning. However, the preferred model depends on the intended clinical operating point: Custom CNN is better if recall/F1 is prioritized, while ResNet50 is better if high precision and threshold-independent ranking are prioritized.

## 9. Confusion matrix analysis

![Custom CNN confusion matrix](../results/figures/custom_cnn_confusion_matrix.png)

![ResNet50 confusion matrix](../results/figures/resnet50_confusion_matrix.png)

The confusion matrices reveal different error profiles:

| Model | False positives | False negatives | Interpretation |
|---|---:|---:|---|
| Logistic Regression | 6441 | 6187 | Many errors in both classes; weak baseline. |
| Custom CNN | 1702 | 3327 | Better tumour detection; fewer missed tumour patches than ResNet50. |
| ResNet50 | 787 | 4888 | Fewer false positives, but more false negatives at threshold 0.5. |

Clinically, false negatives are especially concerning because they represent tumour-containing patches predicted as normal. The Custom CNN missed **3327** tumour patches, while ResNet50 missed **4888**. ResNet50 was more conservative: it produced fewer false positives (**787**) than Custom CNN (**1702**), but this came at the cost of lower recall.

## 10. ROC curve analysis

![Custom CNN ROC curve](../results/figures/custom_cnn_roc_curve.png)

![ResNet50 ROC curve](../results/figures/resnet50_roc_curve.png)

ROC-AUC summarizes threshold-independent ranking. ResNet50 obtained the highest test ROC-AUC (**0.929**), which suggests that its probability scores separate tumour and normal patches slightly better across thresholds than the Custom CNN (**0.921**).

This is important because the decision threshold can be tuned for clinical use. For example, a screening tool may lower the threshold to increase recall, while a triage tool may choose a threshold that balances false positives and false negatives according to workload constraints.

## 11. Ablation studies

Three ablation dimensions were evaluated: augmentation, learning rate, and freezing strategy. Each ablation was run across three seeds: 42, 123, and 2024. This provides a basic estimate of variability and makes the experimental analysis more robust.

### 11.1 Augmentation ablation

| Setting | Accuracy | Precision | Recall | F1 | ROC-AUC | NLL |
|---|---:|---:|---:|---:|---:|---:|
| Off | 0.798 ± 0.008 | 0.912 ± 0.012 | 0.659 ± 0.025 | 0.765 ± 0.014 | 0.911 ± 0.007 | 0.455 ± 0.017 |
| On | 0.818 ± 0.012 | 0.918 ± 0.009 | 0.698 ± 0.034 | 0.793 ± 0.019 | 0.915 ± 0.007 | 0.463 ± 0.025 |


![Augmentation ablation: example ROC and confusion outputs](../results/ablations/augmentation/figures/aug-True_seed123_roc_curve.png)

Interpretation:

- Augmentation improved mean accuracy from **0.798 ± 0.008** to **0.818 ± 0.012**.
- Mean F1 improved from **0.765 ± 0.014** to **0.793 ± 0.019**.
- Mean ROC-AUC changed only slightly, from **0.911 ± 0.007** to **0.915 ± 0.007**.
- This suggests that augmentation mainly improved threshold-level classification behaviour and recall/F1 balance, while global ranking quality remained similar.

### 11.2 Learning-rate ablation

| Setting | Accuracy | Precision | Recall | F1 | ROC-AUC | NLL |
|---|---:|---:|---:|---:|---:|---:|
| 1e-5 | 0.806 ± 0.006 | 0.888 ± 0.003 | 0.701 ± 0.017 | 0.784 ± 0.010 | 0.908 ± 0.002 | 0.416 ± 0.008 |
| 1e-4 | 0.818 ± 0.012 | 0.918 ± 0.009 | 0.698 ± 0.034 | 0.793 ± 0.019 | 0.915 ± 0.007 | 0.463 ± 0.025 |
| 1e-3 | 0.822 ± 0.006 | 0.928 ± 0.011 | 0.698 ± 0.024 | 0.797 ± 0.011 | 0.926 ± 0.005 | 0.420 ± 0.019 |


Interpretation:

- The best mean ROC-AUC was obtained with learning rate **1e-3**: **0.926 ± 0.005**.
- Learning rate **1e-5** was weakest overall, suggesting underfitting within the 5-epoch ablation budget.
- Learning rate **1e-4** was stable but did not outperform 1e-3 in the ablation setting.

### 11.3 Freezing-strategy ablation

| Setting | Accuracy | Precision | Recall | F1 | ROC-AUC | NLL |
|---|---:|---:|---:|---:|---:|---:|
| all_but_fc (linear probe) | 0.798 ± 0.003 | 0.818 ± 0.002 | 0.767 ± 0.009 | 0.791 ± 0.004 | 0.877 ± 0.001 | 0.450 ± 0.002 |
| last_block | 0.818 ± 0.012 | 0.918 ± 0.009 | 0.698 ± 0.034 | 0.793 ± 0.019 | 0.915 ± 0.007 | 0.463 ± 0.025 |
| none (full fine-tune) | 0.863 ± 0.012 | 0.939 ± 0.005 | 0.776 ± 0.021 | 0.850 ± 0.015 | 0.947 ± 0.008 | 0.353 ± 0.046 |


Interpretation:

- Full fine-tuning (`none`) produced the best mean accuracy (**0.863 ± 0.012**), F1 (**0.850 ± 0.015**), ROC-AUC (**0.947 ± 0.008**), and NLL (**0.353 ± 0.046**).
- Freezing all layers except the final classifier performed worse, confirming that ImageNet features alone are not enough for histopathology.
- Fine-tuning only the last block improved over a linear probe, but full fine-tuning was clearly strongest in the ablation results.

Overall, the ablations support three design conclusions: use augmentation, avoid too-low learning rates, and fine-tune more of the ResNet50 backbone when compute allows.

## 12. Robustness under rotations and flips

Histopathology images do not have a fixed natural orientation. Therefore, the final ResNet50 model was evaluated under the eight transformations of the D4 group: identity, rotations, flips, and flip+rotation combinations.

| Transform | Accuracy | Precision | Recall | F1 | ROC-AUC | NLL | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| identity | 0.827 | 0.936 | 0.702 | 0.802 | 0.929 | 0.428 | 15604 | 787 | 4888 | 11489 |
| rot90 | 0.826 | 0.934 | 0.701 | 0.801 | 0.927 | 0.431 | 15581 | 810 | 4889 | 11488 |
| rot180 | 0.830 | 0.935 | 0.709 | 0.806 | 0.931 | 0.422 | 15589 | 802 | 4771 | 11606 |
| rot270 | 0.819 | 0.934 | 0.686 | 0.791 | 0.925 | 0.448 | 15598 | 793 | 5136 | 11241 |
| flip_h | 0.828 | 0.935 | 0.705 | 0.804 | 0.930 | 0.424 | 15589 | 802 | 4826 | 11551 |
| flip_v | 0.820 | 0.931 | 0.690 | 0.793 | 0.927 | 0.440 | 15557 | 834 | 5079 | 11298 |
| flip_h_rot90 | 0.823 | 0.933 | 0.695 | 0.797 | 0.924 | 0.447 | 15577 | 814 | 4991 | 11386 |
| flip_v_rot90 | 0.828 | 0.933 | 0.706 | 0.804 | 0.929 | 0.426 | 15558 | 833 | 4815 | 11562 |


![Rotation and flip robustness](../results/figures/rotation_robustness_plot.png)

Summary statistics across all eight transformations:

| Metric | Mean | Std | Range |
|---|---:|---:|---:|
| Accuracy | 0.825 | 0.004 | 0.011 |
| F1 | 0.800 | 0.006 | 0.015 |
| ROC-AUC | 0.928 | 0.002 | 0.007 |

Interpretation:

- ROC-AUC remained stable across transformations, with a small range of **0.007**.
- Accuracy varied by approximately **0.011** across transformations.
- The model is reasonably robust to rotations and flips, but it is not perfectly rotation-equivariant by architecture.
- This supports the value of augmentation, while also motivating future work with rotation-equivariant CNNs.

## 13. Explainability using Grad-CAM

Grad-CAM was used to visualize the regions that contributed most strongly to ResNet50 tumour predictions.

![Grad-CAM examples](../results/figures/resnet50_gradcam_examples.png)

Interpretation:

- In true-positive examples, Grad-CAM highlights localized tissue regions that increase tumour probability.
- In true-negative examples, the activation is generally weaker or absent, suggesting that the model suppresses tumour evidence on normal patches.
- Some negative examples still show localized activation, which indicates possible attention to artefacts, dense tissue, edges, or visually tumour-like structures.
- Grad-CAM improves interpretability but should not be treated as a complete explanation of clinical reasoning.

For clinical decision support, Grad-CAM can help pathologists review why a patch was flagged, but final interpretation must remain with qualified experts.

## 14. Clinical workflow relevance

A patch-level PCam classifier could be used as one component in a digital pathology workflow:

1. A lymph-node slide is scanned into a whole-slide image.
2. The slide is divided into smaller patches.
3. The model assigns tumour probability scores to patches.
4. High-scoring regions are highlighted for review.
5. A pathologist reviews the highlighted regions and makes the final diagnosis.

Potential benefits:

- faster screening of large slides;
- prioritization of suspicious areas;
- reduced repetitive workload;
- improved consistency in patch-level review;
- interpretable heatmaps for human-in-the-loop review.

The system should be used only as **decision support**, not as autonomous diagnosis.

## 15. Limitations and dataset bias

1. **Patch-level scope.** The project classifies isolated 96×96 patches. Real clinical diagnosis requires whole-slide and patient-level context.
2. **Dataset simplification.** PCam is easier than raw whole-slide pathology workflows because patches are already extracted.
3. **External validation missing.** Performance may decrease on other hospitals, scanners, staining protocols, or patient populations.
4. **Domain shift.** Histopathology images vary due to staining, scanner settings, tissue preparation, and lab protocols.
5. **Threshold sensitivity.** ResNet50 has high precision but lower recall at threshold 0.5. Clinical deployment would require threshold tuning.
6. **Grad-CAM limitations.** Heatmaps are approximate and can highlight correlated artefacts, not necessarily causal diagnostic evidence.
7. **Compute budget.** Main final test metrics are single-checkpoint outputs, while ablations provide multi-seed variability. More full-data multi-seed runs would strengthen confidence intervals.

## 16. Ethics

The model should be positioned as a clinical decision-support tool. Ethical concerns include:

- **False negatives:** missed tumour patches could delay detection or contribute to under-staging.
- **False positives:** excessive false alarms may increase workload and reduce trust.
- **Automation bias:** clinicians may over-trust model outputs if explanations look convincing.
- **Dataset bias:** performance on PCam may not reflect performance in all hospitals or demographic groups.
- **Transparency:** users should know the model was trained on public benchmark data and has not been prospectively validated.
- **Regulation:** real deployment would require clinical validation, quality monitoring, and regulatory approval.

The safest clinical framing is: **AI highlights suspicious regions; pathologists make the final decision.**

## 17. Discussion

The project demonstrates a clear progression from classical ML to deep learning. Logistic Regression underperformed because histopathology classification requires nonlinear visual feature learning. The Custom CNN learned task-specific features from scratch and produced the strongest thresholded performance. ResNet50 showed stronger precision and ROC-AUC, suggesting better probability ranking and more conservative tumour predictions.

The ablation studies add useful design evidence. Augmentation improved classification balance, full fine-tuning outperformed partial freezing, and a higher learning rate was beneficial in the limited ablation setting. The robustness results show that ResNet50 is fairly stable under rotations and flips, though not perfectly invariant. Grad-CAM adds qualitative interpretability and makes the system more suitable for discussion as a clinical decision-support prototype.

A key practical conclusion is that model choice depends on use case. If the goal is to minimize missed tumour patches, the Custom CNN or a recall-optimized ResNet50 threshold may be preferred. If the goal is to reduce false alarms and rank suspicious patches, ResNet50 is attractive due to high precision and ROC-AUC.

## 18. Conclusion

This project built and evaluated a complete PCam metastasis detection pipeline with preprocessing, augmentation, baselines, transfer learning, ablations, robustness testing, and Grad-CAM explainability.

Main conclusions:

- Deep learning strongly outperformed the classical Logistic Regression baseline.
- Custom CNN achieved the best test accuracy, recall, and F1-score.
- ResNet50 achieved the best test precision and ROC-AUC.
- Augmentation improved classification performance in the ablation study.
- Full fine-tuning was the strongest transfer-learning strategy.
- Rotation/flip robustness was reasonably stable but not perfect.
- Grad-CAM provided useful qualitative insight into model focus regions.

Overall, the project supports the potential of AI-assisted patch-level metastasis detection, while also highlighting the need for whole-slide validation, external testing, calibration, and responsible clinical integration.

## 19. Future work

Recommended extensions:

- tune classification thresholds to optimize clinical recall or precision;
- add calibration analysis and reliability diagrams;
- evaluate on external pathology datasets;
- move from patch-level classification to whole-slide aggregation;
- explore multiple-instance learning for slide-level diagnosis;
- test rotation-equivariant CNNs inspired by the PCam paper;
- add stain normalization or domain adaptation;
- build a simple clinical review interface with patch score and Grad-CAM overlay.

## 20. References

1. Veeling, B. S., Linmans, J., Winkens, J., Cohen, T., & Welling, M. **Rotation Equivariant CNNs for Digital Pathology**.
2. CAMELYON16 Challenge.
3. Selvaraju, R. R., et al. **Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization**.
4. He, K., Zhang, X., Ren, S., & Sun, J. **Deep Residual Learning for Image Recognition**.

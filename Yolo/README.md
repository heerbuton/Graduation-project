# Final Package: Recall-Oriented Inference Protocol

## 1. Document Scope
本目录给出一套可直接复现的 `召回优先` 推理方案，目标是在 `Golden_Test_Patches` 上复现实验结果，并提供完整的脚本、权重与结果记录。

本方案仅覆盖**识别阶段（inference + fusion）**，不包含训练过程。

## 2. Fixed Experimental Constraints
为保证与既有实验对齐，本方案遵循以下固定条件：

- 不修改基础训练数据内容。
- 不修改 `Golden_Test_Patches` 的图像与标签。
- 使用既定 8 模型集合进行融合。
- 评价指标采用 micro-level `Precision / Recall / F1`。

## 3. Directory Specification
`final` 目录结构及用途如下：

- `scripts/evaluate_sahi_kfold_vote_fusion.py`：主评估脚本（SAHI 推理 + 多模型融合 + 评估）。
- `scripts/run_recall_first.ps1`：一键复现脚本（已固化召回优先参数）。
- `weights/.../best.pt`：召回优先方案使用的 8 个模型权重。
- `reports/overall_micro_metrics.txt`：总体 micro 指标与关键配置。
- `reports/micro_summary.csv`：总体 micro 指标表格化结果。
- `reports/metrics_summary.csv`：类别级指标（class-wise）统计。

## 4. Model Ensemble Composition
本方案共使用 8 个模型：

1. `weights/y11_v08_kfold_full_results/y11_v08_kfold_fold1/weights/best.pt`
2. `weights/y11_v08_kfold_full_results/y11_v08_kfold_fold2/weights/best.pt`
3. `weights/y11_v08_kfold_full_results/y11_v08_kfold_fold3/weights/best.pt`
4. `weights/y11_v08_kfold_full_results/y11_v08_kfold_fold4/weights/best.pt`
5. `weights/y11_v08_kfold_full_results/y11_v08_kfold_fold5/weights/best.pt`
6. `weights/y11_v08_ultimate_2_1_results/y11_v08_ultimate_2_1/weights/best.pt`
7. `weights/y11_v08_ultimate_23_results/y11_v08_ultimate_23/weights/best.pt`
8. `weights/y11_v10_results/y11_v10_reasonable_aug/weights/best.pt`

## 5. Inference and Fusion Protocol
评估脚本执行流程如下：

1. 对每个模型执行 SAHI 切片推理，收集候选框。
2. 按类别进行投票融合（vote fusion）。
3. 对融合结果执行最终 NMS。
4. 与 Golden 标签匹配并统计 TP / FP / FN。
5. 输出 micro 与 class-wise 指标文件。

默认推理参数（脚本内置）包括：

- `confidence-threshold=0.10`
- `slice-size=1024`
- `overlap-ratio=0.2`
- `postprocess-type=NMS`
- `postprocess-match-metric=IOU`
- `postprocess-match-threshold=0.7`
- `iou-match-threshold=0.5`

## 6. Recall-Oriented Hyperparameters
召回优先方案在上述基础上固定以下融合参数：

- `vote_iou_threshold=0.6`
- `min_votes=2`
- `single_keep_score=0.1`
- `final_nms_iou=0.55`

解释：

- 较高 `vote_iou_threshold` 保持同类框聚类一致性。
- `min_votes=2` 抑制单模型偶发噪声。
- `single_keep_score=0.1` 允许高分单框保留，以换取更高召回。
- `final_nms_iou=0.55` 平衡重复框抑制与漏检风险。

## 7. Reproducibility Procedure
### 7.1 Environment

- Python 环境：`F:\anaconda\envs\pytorch\python.exe`
- 项目根目录：`F:\AIcharacter\YOLO11`

### 7.2 One-command Reproduction
在 PowerShell 执行：

```powershell
cd F:\AIcharacter\YOLO11\final\scripts
.\run_recall_first.ps1
```

脚本输出目录：

- `F:\AIcharacter\YOLO11\Golden_Test_Patches\sahi_results_vote_8models_recall_first_from_final`

### 7.3 Custom Runtime Arguments
可覆盖默认 Python 与项目路径：

```powershell
.\run_recall_first.ps1 -PythonExe "F:\anaconda\envs\pytorch\python.exe" -ProjectRoot "F:\AIcharacter\YOLO11"
```

## 8. Evaluation Metrics Definition
采用 micro-level 定义：

- `Precision = TP / (TP + FP)`
- `Recall = TP / (TP + FN)`
- `F1 = 2 * Precision * Recall / (Precision + Recall)`

其中 TP、FP、FN 为所有类别汇总统计量。

## 9. Recorded Baseline Result (Recall-first)
当前归档结果（见 `reports/overall_micro_metrics.txt`）：

- `TP=4392, FP=3255, FN=1023`
- `Precision=0.5743`
- `Recall=0.8111`
- `F1=0.6725`

该配置对应“召回优先”目标，即优先提高 `Recall`，并接受一定 `Precision` 下降。

## 10. Integrity and Change Control
若替换以下任一要素，应视为新实验版本并重新记录结果：

- `weights` 下任一 `best.pt`
- `scripts/evaluate_sahi_kfold_vote_fusion.py`
- `run_recall_first.ps1` 中融合参数
- Golden 数据目录内容

建议每次变更后至少更新：

- `reports/overall_micro_metrics.txt`
- `reports/micro_summary.csv`
- 版本说明（本 README 的结果段落）

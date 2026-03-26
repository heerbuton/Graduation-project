param(
  [string]$PythonExe = "F:\anaconda\envs\pytorch\python.exe",
  [string]$ProjectRoot = "F:\AIcharacter\YOLO11"
)

$FinalRoot = Split-Path -Parent $PSScriptRoot
$WeightsRoot = Join-Path $FinalRoot 'weights'
$ScriptPath = Join-Path $PSScriptRoot 'evaluate_sahi_kfold_vote_fusion.py'
$OutDir = Join-Path $ProjectRoot 'Golden_Test_Patches\sahi_results_vote_8models_recall_first_from_final'

& $PythonExe $ScriptPath `
  --kfold-root (Join-Path $WeightsRoot 'y11_v08_kfold_full_results') `
  --extra-model (Join-Path $WeightsRoot 'y11_v08_ultimate_2_1_results\y11_v08_ultimate_2_1\weights\best.pt') `
  --extra-model (Join-Path $WeightsRoot 'y11_v08_ultimate_23_results\y11_v08_ultimate_23\weights\best.pt') `
  --extra-model (Join-Path $WeightsRoot 'y11_v10_results\y11_v10_reasonable_aug\weights\best.pt') `
  --images-dir (Join-Path $ProjectRoot 'Golden_Test_Patches\images') `
  --labels-dir (Join-Path $ProjectRoot 'Golden_Test_Patches\labels') `
  --output-dir $OutDir `
  --vote-iou-threshold 0.6 `
  --min-votes 2 `
  --single-keep-score 0.1 `
  --final-nms-iou 0.55

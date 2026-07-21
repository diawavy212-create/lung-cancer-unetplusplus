$ErrorActionPreference = "Stop"

$Python = "..\.venv\Scripts\python.exe"
$TrainImages = "..\data\dataset_liver_png\train_images"
$TrainMasks = "..\data\dataset_liver_png\train_masks"
$ValImages = "..\data\dataset_liver_png\val_images"
$ValMasks = "..\data\dataset_liver_png\val_masks"

& $Python -m liver_cancer_ai.train `
  --train-images $TrainImages `
  --train-masks $TrainMasks `
  --val-images $ValImages `
  --val-masks $ValMasks `
  --epochs 50 `
  --batch-size 2 `
  --image-size 320 `
  --model unetpp `
  --base-channels 16 `
  --lr 0.0003 `
  --tumor-weight 5 `
  --loss dice-ce `
  --deep-supervision `
  --amp `
  --device cuda `
  --output "..\data\paper_unetpp_320_50ep_best.pth" `
  --history-csv "..\data\paper_unetpp_320_50ep_history.csv" `
  --plot-output "..\data\paper_unetpp_320_50ep_curves.png"

& $Python -m liver_cancer_ai.tune_threshold `
  --weights "..\data\paper_unetpp_320_50ep_best.pth" `
  --val-images $ValImages `
  --val-masks $ValMasks `
  --min-threshold 0.60 `
  --max-threshold 0.99 `
  --step 0.01 `
  --output-csv "..\data\paper_unetpp_320_50ep_thresholds.csv" `
  --plot-output "..\data\paper_unetpp_320_50ep_thresholds.png" `
  --device cuda

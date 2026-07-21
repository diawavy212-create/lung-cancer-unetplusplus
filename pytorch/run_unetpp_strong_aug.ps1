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
  --batch-size 4 `
  --image-size 256 `
  --model unetpp `
  --base-channels 16 `
  --lr 0.0003 `
  --tumor-weight 5 `
  --loss dice-ce `
  --deep-supervision `
  --strong-augment `
  --amp `
  --device cuda `
  --output "..\data\paper_unetpp_256_50ep_strong_aug_best.pth" `
  --history-csv "..\data\paper_unetpp_256_50ep_strong_aug_history.csv" `
  --plot-output "..\data\paper_unetpp_256_50ep_strong_aug_curves.png"

& $Python -m liver_cancer_ai.tune_threshold `
  --weights "..\data\paper_unetpp_256_50ep_strong_aug_best.pth" `
  --val-images $ValImages `
  --val-masks $ValMasks `
  --min-threshold 0.60 `
  --max-threshold 0.99 `
  --step 0.01 `
  --output-csv "..\data\paper_unetpp_256_50ep_strong_aug_thresholds.csv" `
  --plot-output "..\data\paper_unetpp_256_50ep_strong_aug_thresholds.png" `
  --device cuda

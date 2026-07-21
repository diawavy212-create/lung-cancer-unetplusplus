$ErrorActionPreference = "Stop"

$Python = "..\.venv\Scripts\python.exe"
$TrainImages = "..\data\dataset_liver_png\train_images"
$TrainMasks = "..\data\dataset_liver_png\train_masks"
$ValImages = "..\data\dataset_liver_png\val_images"
$ValMasks = "..\data\dataset_liver_png\val_masks"

$Experiments = @(
    @{ Model = "unet"; Name = "paper_unet_256_50ep" },
    @{ Model = "unetpp"; Name = "paper_unetpp_256_50ep" },
    @{ Model = "eca-unetpp"; Name = "paper_eca_unetpp_256_50ep" },
    @{ Model = "eca-dilated-unetpp"; Name = "paper_eca_dilated_unetpp_256_50ep" }
)

foreach ($Exp in $Experiments) {
    Write-Host "==== Training $($Exp.Model) ===="
    & $Python -m liver_cancer_ai.train `
        --train-images $TrainImages `
        --train-masks $TrainMasks `
        --val-images $ValImages `
        --val-masks $ValMasks `
        --epochs 50 `
        --batch-size 4 `
        --image-size 256 `
        --model $Exp.Model `
        --base-channels 16 `
        --lr 0.0003 `
        --tumor-weight 8 `
        --loss dice-ce `
        --deep-supervision `
        --amp `
        --device cuda `
        --output "..\data\$($Exp.Name)_best.pth" `
        --history-csv "..\data\$($Exp.Name)_history.csv" `
        --plot-output "..\data\$($Exp.Name)_curves.png"
}

Write-Host "All comparison experiments finished."

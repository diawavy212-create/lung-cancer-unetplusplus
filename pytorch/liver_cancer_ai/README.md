# Liver Cancer CT Diagnosis

This module adds the requested product algorithm stack:

- PyTorch
- UNet++ nested skip connections
- ECA channel attention
- dilated convolutions
- Flask upload inference API

## Recommended datasets

Use one of these public liver CT segmentation datasets:

- LiTS Liver Tumor Segmentation Challenge: best match for liver tumor CT segmentation. It includes liver and tumor annotations and is commonly used for hepatocellular carcinoma / liver lesion segmentation research.
- Medical Segmentation Decathlon Task03 Liver: derived from the liver tumor segmentation benchmark and already matches the original UNet++ repository's liver task notes.
- TCIA collections: useful when you need additional DICOM liver cancer CT data, but labels may require extra preprocessing or manual annotation depending on the collection.

For this project's current `train.py`, convert the dataset into paired 2D image/mask folders:

```text
dataset/
  train_images/
    case_0001_slice_050.png
  train_masks/
    case_0001_slice_050.png
  val_images/
    case_0002_slice_040.png
  val_masks/
    case_0002_slice_040.png
```

Mask convention: background is `0`, tumor region is any positive value.

LiTS/MSD are usually distributed as 3D NIfTI volumes. Convert them with:

```bash
cd pytorch
python -m liver_cancer_ai.prepare_liver_dataset ^
  --images path/to/imagesTr ^
  --labels path/to/labelsTr ^
  --output dataset_liver_png
```

The converter keeps tumor-positive slices by default. In LiTS/MSD labels, `0` is background, `1` is liver, and `2` is tumor.

## Run inference

```bash
cd pytorch
python -m liver_cancer_ai.inference path/to/ct.png --weights path/to/model.pth
```

If `--weights` is omitted, the network runs with random weights. That is useful only for checking the software path; clinical prediction requires a trained liver CT checkpoint.

## Run the web API

```bash
cd pytorch
set LIVER_MODEL_WEIGHTS=path\to\model.pth
python -m liver_cancer_ai.api
```

Then open `http://127.0.0.1:5000`.

The API accepts `POST /api/predict` with form-data key `file` and returns:

- diagnosis
- prediction_time
- confidence
- lesion_ratio
- accuracy
- dice

`accuracy` and `dice` are returned as `null` during single-image inference because they require a labeled validation set or ground-truth mask.

## Train with labeled masks

Prepare paired image and mask folders with matching file names:

```text
dataset/
  train_images/
  train_masks/
  val_images/
  val_masks/
```

Then run:

```bash
cd pytorch
python -m liver_cancer_ai.train ^
  --train-images dataset/train_images ^
  --train-masks dataset/train_masks ^
  --val-images dataset/val_images ^
  --val-masks dataset/val_masks ^
  --epochs 50 ^
  --output liver_eca_unetpp_best.pth
```

The training loop reports validation Dice and pixel accuracy each epoch, and saves the best checkpoint for the API.

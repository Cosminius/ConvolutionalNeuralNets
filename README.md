# Convolutional Neural Networks — PyTorch

Code and notes from **Chapter 12 – Deep Computer Vision Using Convolutional Neural Networks** of
*Hands-On Machine Learning with Scikit-Learn and PyTorch* (Aurélien Géron, 2025).

Everything is written in **PyTorch / torchvision**, running locally on an Apple Silicon GPU (`device = "mps"`).

---

## Contents

### `CNNs-chatper12.ipynb`
The main notebook, working through the whole chapter:

**1. Convolutional layers from scratch**
- Loading the sample images from `sklearn.datasets.load_sample_images()`, scaling to `[0, 1]` and permuting to PyTorch's `(N, C, H, W)` layout
- `T.CenterCrop` for cropping, plotting the images with matplotlib
- `nn.Conv2d` with random filters — inspecting feature-map shapes, the effect of `padding="same"`, and the shapes of the `weight` and `bias` tensors

**2. Pooling layers**
- `nn.MaxPool2d`
- **`DepthPool`** — a custom implementation of *depthwise max pooling* (pooling along the channel axis) using `F.max_pool1d`
- Global average pooling: `nn.AdaptiveAvgPool2d(1)` vs. the manual equivalent `mean(dim=(2, 3))`

**3. A CNN from scratch on Fashion-MNIST**
- Classic VGG-style architecture: `Conv → ReLU → MaxPool` blocks with a doubling filter count (64 → 128 → 256), followed by a dense head with `Dropout`
- `DefaultConv2d = partial(nn.Conv2d, kernel_size=3, padding="same")` to cut down on repetition
- A 55,000 / 5,000 / 10,000 (train / valid / test) split via `random_split`
- A custom training loop (`train`) plus an evaluation function (`evaluate_tm`), using `torchmetrics.Accuracy`, `CrossEntropyLoss` and `AdamW`

**4. Modern architectures**
- **`SeparableConv2d`** — depthwise separable convolution (depthwise + pointwise), the building block of Xception
- **`ResidualUnit`** and **`ResNet34`** — see the dedicated notebook below

**5. Pretrained models**
- `convnext_base` with `IMAGENET1K_V1` weights
- Automatic preprocessing through `weights.transforms()`, top-3 predictions with `topk` + `softmax`, class-id lookup via `weights.meta["categories"]`

**6. Transfer learning on Flowers-102**
- Replacing the classification head (`model.classifier[2]`) with a 102-class `nn.Linear`
- Two-phase training: first with the **backbone frozen** (`requires_grad = False`), then with the **whole model unfrozen** for fine-tuning
- Data augmentation: `RandomHorizontalFlip`, `RandomRotation`, `RandomResizedCrop`, `ColorJitter`, `Normalize`

**7. Classification + localization**
- **`FlowerLocator`** — a ConvNeXt model with two heads: one for classification and one that regresses a bounding box (4 coordinates)
- `torchvision.tv_tensors.BoundingBoxes` — bounding boxes that are transformed automatically together with the image (`CXCYWH` format)
- Visualization helpers: `get_image_with_bbox()` (`draw_bounding_boxes` + `box_convert`) and `denormalize()` to undo the ImageNet normalization
- **`FlowersWithBbox`** — a custom `Dataset` returning the image plus `{"label", "bbox"}`

**8. Object detection**
- **YOLOv9** through `ultralytics`: inference on images and **tracking** on video (`model.track(..., stream=True)`), reading the per-frame `track_id`s

**9. Semantic segmentation**
- Pretrained `fcn_resnet50` — extracting the `person` class mask from the softmax output and overlaying it on the original image

**10. The mAP metric**
- `maximum_precisions()` — the reverse cumulative maximum precision, the precision/recall curve, and the **mAP** computation

### `ResNet-34.ipynb`
ResNet-34 implemented from scratch:
- **`ResidualUnit`** — two `Conv → BatchNorm → ReLU` layers on the main path, plus a skip connection with a `1×1` convolution whenever `stride > 1` (to match the dimensions)
- **`ResNet34`** — a `7×7 / stride 2` stem + `MaxPool`, then the `[64]*3 + [128]*4 + [256]*6 + [512]*3` blocks, global average pooling, and the final dense layer

---

## Exercises (Chapter 12)

### `CNN-MNIST.ipynb` — a CNN from scratch on MNIST
> *Exercise 9: "Build your own CNN from scratch and try to achieve the highest possible accuracy on MNIST."*

**Result: ~99.4% accuracy on the test set.**

- **Data**: MNIST (`torchvision.datasets`), normalized with the dataset's mean/std (`0.1307` / `0.3081`);
  a 55,000 / 5,000 `random_split` of the training set for validation (fixed seed, `torch.manual_seed(23)`), plus the 10,000 test images
- **`MNISTClassifier` architecture** — 4 `Conv2d(3×3, padding="same") → BatchNorm2d → LeakyReLU` blocks
  with a doubling filter count (16 → 32 → 64 → 128) and a `MaxPool2d(2)` after every pair,
  then **global average pooling** (`AdaptiveAvgPool2d(1)`) and a `128 → 256 → 128 → 10` dense head
  with `LeakyReLU` between the layers (without activations, the three `Linear` layers would mathematically collapse into one)
- **Training** — `AdamW` + `CrossEntropyLoss`, `torchmetrics.Accuracy`, 20 epochs (~10 s/epoch on MPS)
- **`train_with_early_stopping_lr_scheduler_warm_up()`** — a custom training loop combining three mechanisms:
  - **warm-up** over the first 3 epochs (`LambdaLR`, learning rate ramping from 10% to 100%)
  - **`ReduceLROnPlateau`** (`mode="max"`, `patience=2`, `factor=0.1`) after warm-up — the jump off the plateau
    (`valid: 0.9814 → 0.9938`) right after the learning rate drop is clearly visible
  - **early stopping** with checkpointing: the `state_dict` is saved every time validation accuracy improves,
    and the best model is reloaded at the end
- **Plots** — the training loss and the train vs. validation curves, showing the effect of the learning rate drop

### `transfer_learning.ipynb` — transfer learning on Hymenoptera (bees vs. ants)
> *Exercise 10: "Use transfer learning for large image classification" (the dataset from the official PyTorch tutorial).*

**Result: 100% on both validation and test**, after 7 epochs of feature extraction + 6 of fine-tuning.

- **Data**: `hymenoptera_data` loaded with `ImageFolder` (244 training images, 153 validation);
  those 153 are split further into **70 validation / 83 test** with `random_split`
- **Preprocessing pipeline** (`torchvision.transforms.v2`), different for training and evaluation:
  - train: `RandomResizedCrop(224)` + `RandomHorizontalFlip` (data augmentation)
  - val/test: `Resize(256)` + `CenterCrop(224)`
  - both: normalization with the ImageNet statistics (`[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`)
- **Model**: pretrained `convnext_small` (`ConvNeXt_Small_Weights.IMAGENET1K_V1`),
  with `model.classifier[2]` replaced by an `nn.Linear(768, 2)`
- **Two-phase training**, using the same loop as MNIST (warm-up + `ReduceLROnPlateau` + early stopping with `patience=5`):
  1. **Feature extraction** — the whole backbone frozen (`requires_grad = False`), only the classification head
     is trained; `AdamW` with the default learning rate, ~6 s/epoch on MPS
  2. **Fine-tuning** — the whole model is unfrozen and training continues with a **small learning rate (`1e-5`)**,
     so the representations learned on ImageNet are not destroyed; ~18 s/epoch

> **Why 100%?** There is no leakage between the splits (training lives in a separate folder, and validation and test
> come out of the same `random_split`, so they cannot overlap). The task is simply easy: ImageNet already contains ant
> and bee classes, so the backbone arrives with almost ready-made features, and the evaluation sets are tiny
> (70 and 83 images respectively) — a single mistake would already cost ~1.2%. The fine-tuning phase has essentially
> nothing left to improve; it only shows that the model does not degrade.

> `hymenoptera_data` is not downloaded automatically — grab it
> [here](https://download.pytorch.org/tutorial/hymenoptera_data.zip) and unzip it into `datasets/hymenoptera/`
> (structure: `train/{ants,bees}` and `val/{ants,bees}`).

---

## Dependencies

```bash
pip install torch torchvision torchmetrics scikit-learn matplotlib numpy ultralytics
```

## Running

```bash
jupyter lab
```

The datasets (Fashion-MNIST, MNIST, Flowers-102) are downloaded automatically into `datasets/`,
and the YOLO weights (`yolov9m.pt`) on first run — which is why they are not committed to the repo.
The one exception is `hymenoptera_data`, which has to be downloaded manually (see above).

> The notebooks use `device = "mps"` (Apple Silicon). On other hardware, change it to `"cuda"` or `"cpu"`.

## What's in the repo

Only the notebooks and this README. Datasets, model weights, videos,
the YOLO outputs in `runs/` and IDE files are excluded via `.gitignore`.

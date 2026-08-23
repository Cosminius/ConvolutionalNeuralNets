# Convolutional Neural Networks — PyTorch

Cod si notite de la **Capitolul 12 – Deep Computer Vision Using Convolutional Neural Networks** din
*Hands-On Machine Learning with Scikit-Learn and PyTorch* (Aurélien Géron, 2025).

Totul este scris in **PyTorch / torchvision**, rulat local pe GPU Apple Silicon (`device = "mps"`).

---

## Continut

### `CNNs.ipynb`
Notebook-ul principal, care parcurge tot capitolul:

**1. Straturi convolutionale de la zero**
- Incarcarea imaginilor de test din `sklearn.datasets.load_sample_images()`, normalizare in `[0, 1]` si permutarea la formatul PyTorch `(N, C, H, W)`
- `T.CenterCrop` pentru decupare, vizualizarea imaginilor cu matplotlib
- `nn.Conv2d` cu filtre aleatoare — inspectarea formei feature-map-urilor, efectul lui `padding="same"`, forma tensorilor `weight` si `bias`

**2. Straturi de pooling**
- `nn.MaxPool2d`
- **`DepthPool`** — implementare proprie de *depthwise max pooling* (pooling pe axa canalelor) folosind `F.max_pool1d`
- Global average pooling: `nn.AdaptiveAvgPool2d(1)` vs. echivalentul manual `mean(dim=(2, 3))`

**3. CNN de la zero pe Fashion-MNIST**
- Arhitectura clasica de tip VGG: blocuri `Conv → ReLU → MaxPool` cu numar dublat de filtre (64 → 128 → 256), urmate de un cap dens cu `Dropout`
- `DefaultConv2d = partial(nn.Conv2d, kernel_size=3, padding="same")` pentru a scapa de repetitie
- Split 55 000 / 5 000 / 10 000 (train / valid / test) cu `random_split`
- Bucla de antrenare proprie (`train`) + functia de evaluare (`evaluate_tm`), cu `torchmetrics.Accuracy`, `CrossEntropyLoss` si `AdamW`

**4. Arhitecturi moderne**
- **`SeparableConv2d`** — convolutie separabila in adancime (depthwise + pointwise), blocul de baza al Xception
- **`ResidualUnit`** si **`ResNet34`** — vezi notebook-ul dedicat de mai jos

**5. Modele pre-antrenate**
- `convnext_base` cu greutati `IMAGENET1K_V1`
- Preprocesare automata prin `weights.transforms()`, predictii top-3 cu `topk` + `softmax`, maparea id-urilor de clase prin `weights.meta["categories"]`

**6. Transfer learning pe Flowers-102**
- Inlocuirea capului de clasificare (`model.classifier[2]`) cu un `nn.Linear` de 102 clase
- Antrenare in doua faze: intai cu **backbone-ul inghetat** (`requires_grad = False`), apoi cu **tot modelul dezghetat** pentru fine-tuning
- Data augmentation: `RandomHorizontalFlip`, `RandomRotation`, `RandomResizedCrop`, `ColorJitter`, `Normalize`

**7. Clasificare + localizare**
- **`FlowerLocator`** — modelul ConvNeXt cu doua capete: unul de clasificare si unul de regresie pentru bounding box (4 coordonate)
- `torchvision.tv_tensors.BoundingBoxes` — bounding box-uri care se transforma automat impreuna cu imaginea (format `CXCYWH`)
- Utilitare de vizualizare: `get_image_with_bbox()` (`draw_bounding_boxes` + `box_convert`) si `denormalize()` pentru a anula normalizarea ImageNet
- **`FlowersWithBbox`** — `Dataset` custom care returneaza imagine + `{"label", "bbox"}`

**8. Object detection**
- **YOLOv9** prin `ultralytics`: inferenta pe imagini si **tracking** pe video (`model.track(..., stream=True)`), cu citirea `track_id`-urilor per frame

**9. Segmentare semantica**
- `fcn_resnet50` pre-antrenat — extragerea mastii pentru clasa `person` din softmax si aplicarea ei peste imaginea originala

**10. Metrica mAP**
- `maximum_precisions()` — precizia maxima cumulata invers, curba precision/recall si calculul **mAP**

### `ResNet-34.ipynb`
Implementarea ResNet-34 de la zero:
- **`ResidualUnit`** — doua straturi `Conv → BatchNorm → ReLU` pe calea principala, plus skip connection cu convolutie `1×1` atunci cand `stride > 1` (pentru a potrivi dimensiunile)
- **`ResNet34`** — stem `7×7 / stride 2` + `MaxPool`, apoi blocurile `[64]*3 + [128]*4 + [256]*6 + [512]*3`, global average pooling si stratul dens final

---

## Dependinte

```bash
pip install torch torchvision torchmetrics scikit-learn matplotlib numpy ultralytics
```

## Rulare

```bash
jupyter lab
```

Seturile de date (Fashion-MNIST, Flowers-102) se descarca automat in `datasets/`,
iar greutatile YOLO (`yolov9m.pt`) la prima rulare — de aceea nu sunt urcate in repo.

> Notebook-urile folosesc `device = "mps"` (Apple Silicon). Pe alt hardware, schimba in `"cuda"` sau `"cpu"`.

## Ce este in repo

Doar notebook-urile si acest README. Datasets, greutati de modele, video-uri,
output-urile YOLO din `runs/` si fisierele de IDE sunt excluse prin `.gitignore`.

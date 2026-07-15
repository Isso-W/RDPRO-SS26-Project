# consensus.md — 数据集特征 → 组件共识

> ⚠ 存在 `traits_verified=False` 的竞赛——特征卡尚未人工核对，以下共识为初判。逐竞赛核对后重跑本表。

> 阈值：support ≥ 0.5，breadth ≥ 2。`passed` 列标记是否达标。

# task_type = classification

## fine_grained — backbone·发动机 engine （KB 覆盖率 74%，本组占全部 backbone 票 74%）

| kb_id | support | breadth | votes/total | passed | raw（归并痕迹） |
|---|---|---|---|---|---|
| efficientnet | 0.48 | 10 | 30.6/63.1 | ✔dom(×1.81,+0.217) | All Data Ext 2020 Efficientnet b6, All Data Ext 2020 Efficie |
| resnet | 0.27 | 10 | 16.9/63.1 |  | Ibn-resnet50, ResNeSt50, ResNeXt, ResNeXt50, SEResNeXt50, Se |
| vit | 0.09 | 3 | 5.4/63.1 |  | DeiT-base-384, ViT, ViT large, Vision Transformer(base patch |
| swin_transformer | 0.07 | 3 | 4.4/63.1 |  | Hybrid SwinBase224-Efficentnet b5, Hybrid SwinBase224-Effice |
| convnext | 0.07 | 5 | 4.3/63.1 |  | ConvNext Base, ConvNext Large, ConvNext XLarge, convnext bas |
| mobilenet_v3 | 0.02 | 3 | 1.5/63.1 |  | CropNet (MobileNetv3), MobileNetv3 large initialized with Cr |

## fine_grained — loss （KB 覆盖率 38%）

| kb_id | support | breadth | votes/total | passed | raw（归并痕迹） |
|---|---|---|---|---|---|
| cross_entropy_loss | 0.63 | 6 | 10.2/16.3 | ✔dom(×1.67,+0.252) | BCE, BCE loss, BCE loss + weight*SupCon loss, Binary Cross‑E |
| focal_loss | 0.37 | 5 | 6.1/16.3 |  | BCE2wayloss(Group1), BCEFocal2WayLoss(Group2), BCEFocal2WayL |

## class_imbalance — backbone·车架 frame （KB 覆盖率 72%，本组占全部 backbone 票 2%）

| kb_id | support | breadth | votes/total | passed | raw（归并痕迹） |
|---|---|---|---|---|---|
| unet | 0.59 | 1 | 2.0/3.4 |  | 3D U-Net with a residual encoder, nnU-Net 3D, nnUNetResEncUN |
| yolov8 | 0.41 | 1 | 1.4/3.4 |  | YOLOv11m, Custom YOLO with timm backbone, YOLOv11x 1280, yol |

## class_imbalance — backbone·发动机 engine （KB 覆盖率 72%，本组占全部 backbone 票 70%）

| kb_id | support | breadth | votes/total | passed | raw（归并痕迹） |
|---|---|---|---|---|---|
| efficientnet | 0.41 | 15 | 41.9/101.5 | ✔dom(×2.3,+0.234) | 3D CenterNet with 2D Effv2s extractor, All Data Ext 2020 Eff |
| convnext | 0.18 | 10 | 18.2/101.5 |  | 3D ConvNeXt, ConvNeXt-base, ConvNeXt-small, ConvNeXt Small,  |
| resnet | 0.17 | 12 | 17.5/101.5 |  | 3D ResNet-18, ResNeSt50, ResNeXt, ResNeXt50, ResNet18, ResNe |
| vit | 0.12 | 8 | 12.5/101.5 |  | DeiT-base-384, SE-ResNext50 + deit-tiny-patch16-224, ViT, Vi |
| swin_transformer | 0.09 | 6 | 9.4/101.5 |  | Hybrid SwinBase224-Efficentnet b5, Hybrid SwinBase224-Effice |
| mobilenet_v3 | 0.01 | 3 | 1.5/101.5 |  | 1D version of MobileNetV2, CropNet (MobileNetv3), mobilenetv |
| dinov2 | 0.00 | 1 | 0.5/101.5 |  | dinov2 vit family, dinov2 vit family, dinov2 vit family, din |

## class_imbalance — loss （KB 覆盖率 35%）

| kb_id | support | breadth | votes/total | passed | raw（归并痕迹） |
|---|---|---|---|---|---|
| cross_entropy_loss | 0.71 | 10 | 18.9/26.5 | ✔dom(×2.66,+0.445) | BCE, BCE loss, BCE loss + weight*SupCon loss, BCELoss(Binary |
| focal_loss | 0.27 | 6 | 7.1/26.5 |  | BCE2wayloss(Group1), BCEFocal2WayLoss(Group2), BCEFocal2WayL |
| bce_dice_loss | 0.02 | 1 | 0.5/26.5 |  | BCELoss * 0.2 + DICELoss * 0.8 |

## medical — backbone·车架 frame （KB 覆盖率 72%，本组占全部 backbone 票 5%）

| kb_id | support | breadth | votes/total | passed | raw（归并痕迹） |
|---|---|---|---|---|---|
| unet | 0.59 | 1 | 2.0/3.4 |  | 3D U-Net with a residual encoder, nnU-Net 3D, nnUNetResEncUN |
| yolov8 | 0.41 | 1 | 1.4/3.4 |  | YOLOv11m, Custom YOLO with timm backbone, YOLOv11x 1280, yol |

## medical — backbone·发动机 engine （KB 覆盖率 72%，本组占全部 backbone 票 68%）

| kb_id | support | breadth | votes/total | passed | raw（归并痕迹） |
|---|---|---|---|---|---|
| efficientnet | 0.32 | 7 | 15.7/48.9 |  | 3D CenterNet with 2D Effv2s extractor, EfficientNet, Efficie |
| convnext | 0.29 | 6 | 14.3/48.9 |  | 3D ConvNeXt, ConvNeXt-base, ConvNeXt-small, ConvNeXt Small,  |
| vit | 0.15 | 5 | 7.1/48.9 |  | SE-ResNext50 + deit-tiny-patch16-224, deit3_small, beitv2_ba |
| swin_transformer | 0.12 | 4 | 6.0/48.9 |  | Swin-Tiny, swin base, swin transformer v2 large, swin_large_ |
| resnet | 0.10 | 5 | 4.8/48.9 |  | 3D ResNet-18, ResNet18, ResNet18, ResNet18, Resnet10t, Resne |
| mobilenet_v3 | 0.01 | 1 | 0.5/48.9 |  | 1D version of MobileNetV2 |
| dinov2 | 0.01 | 1 | 0.5/48.9 |  | dinov2 vit family, dinov2 vit family, dinov2 vit family, din |

## medical — loss （KB 覆盖率 31%）

| kb_id | support | breadth | votes/total | passed | raw（归并痕迹） |
|---|---|---|---|---|---|
| cross_entropy_loss | 0.89 | 6 | 11.7/13.2 | ✔dom(×11.69,+0.811) | BCE, BCE loss, BCELoss(Binary: benign or malignant) + CrossE |
| focal_loss | 0.08 | 1 | 1.0/13.2 |  | Focal loss with alpha=0.75, weighted BCE loss |
| bce_dice_loss | 0.04 | 1 | 0.5/13.2 |  | BCELoss * 0.2 + DICELoss * 0.8 |

## multi_label — backbone·发动机 engine （KB 覆盖率 100%，本组占全部 backbone 票 100%）

| kb_id | support | breadth | votes/total | passed | raw（归并痕迹） |
|---|---|---|---|---|---|
| resnet | 0.50 | 1 | 1.9/3.8 |  | SEResNeXt50, SeResNext50, resnet50, resnext50_32x4d, resnet5 |
| efficientnet | 0.37 | 1 | 1.4/3.8 |  | Effnet B5, efficientnetv2 |
| mobilenet_v3 | 0.13 | 1 | 0.5/3.8 |  | MobileNetv3 large initialized with Crop net weights |

## multi_label — loss （KB 覆盖率 42%）

| kb_id | support | breadth | votes/total | passed | raw（归并痕迹） |
|---|---|---|---|---|---|
| cross_entropy_loss | 1.00 | 1 | 1.0/1.0 |  | BCE loss, cross entropy |

# task_type = image_segmentation

## class_imbalance — backbone·车架 frame （KB 覆盖率 75%，本组占全部 backbone 票 25%）

| kb_id | support | breadth | votes/total | passed | raw（归并痕迹） |
|---|---|---|---|---|---|
| unet | 0.38 | 2 | 2.7/7.2 |  | 3D Encoder 2D Decoder U-Net model using Squeeze-and-Excitati |
| yolov8 | 0.35 | 1 | 2.5/7.2 |  | YOLOX-x, YOLOv5x6, YOLOv7x, YOLOv8l, YOLOv8x, YOLOv8l, Yolov |
| segformer | 0.28 | 1 | 2.0/7.2 |  | 3d unet(16 channels) + segformer b3, 3d unet(16 channels) +  |

## class_imbalance — backbone·发动机 engine （KB 覆盖率 75%，本组占全部 backbone 票 50%）

| kb_id | support | breadth | votes/total | passed | raw（归并痕迹） |
|---|---|---|---|---|---|
| resnet | 0.31 | 3 | 4.3/14.0 |  | 3D SEResnet101, 3D Resnet34, 3D Resnet34, 3D resnet152, 3D r |
| convnext | 0.24 | 3 | 3.3/14.0 |  | Cascade Mask RCNN + Convnext v2 Large, convnext-small, convn |
| efficientnet | 0.19 | 3 | 2.7/14.0 |  | EfficientNetB1-Unet, EfficientNetB2-Unet, efficientnet_b5_Un |
| vit | 0.16 | 2 | 2.2/14.0 |  | ViT-Adapter-L, maxvit_base, maxvit_tiny_tf_512.in1k |
| swin_transformer | 0.11 | 2 | 1.5/14.0 |  | MaskRCNN (swint), swin-t, swinv2_tiny |

## class_imbalance — loss （KB 覆盖率 44%）

| kb_id | support | breadth | votes/total | passed | raw（归并痕迹） |
|---|---|---|---|---|---|
| bce_dice_loss | 0.47 | 2 | 4.0/8.6 |  | BCE + Dice Loss, BCE and hard dice, BCELoss, a mixture of BC |
| focal_loss | 0.30 | 2 | 2.6/8.6 |  | BoundaryLoss + 0.5 Focal Symmetric Loss, FocalLoss, binary-f |
| cross_entropy_loss | 0.23 | 1 | 2.0/8.6 |  | BCE, CrossEntropyLoss with label smoothing 0.3, bce + global |

## medical — backbone·车架 frame （KB 覆盖率 74%，本组占全部 backbone 票 19%）

| kb_id | support | breadth | votes/total | passed | raw（归并痕迹） |
|---|---|---|---|---|---|
| unet | 0.59 | 4 | 5.5/9.4 | ✔dom(×2.2,+0.319) | 3D UNet (MONAI), DynUnet, U-Net3D, UNet (MaxViT-Large 512),  |
| yolov8 | 0.27 | 1 | 2.5/9.4 |  | YOLOX-x, YOLOv5x6, YOLOv7x, YOLOv8l, YOLOv8x, YOLOv8l, Yolov |
| segformer | 0.15 | 2 | 1.4/9.4 |  | SegFormer mit-b3, SegFormer mit-b4, SegFormer mit-b5, SegFor |

## medical — backbone·发动机 engine （KB 覆盖率 74%，本组占全部 backbone 票 55%）

| kb_id | support | breadth | votes/total | passed | raw（归并痕迹） |
|---|---|---|---|---|---|
| efficientnet | 0.34 | 5 | 9.3/27.6 |  | EfficientNet B4 - Unet, EfficientNet B5 - Unet, EfficientNet |
| convnext | 0.26 | 4 | 7.1/27.6 |  | Cascade Mask RCNN + Convnext v2 Large, ConvNeXt T - DeeplabV |
| resnet | 0.21 | 4 | 5.7/27.6 |  | CascadeRCNN (ResNeXt, Regnet), Detectors ResNeXt-101-32x4d,  |
| swin_transformer | 0.12 | 4 | 3.3/27.6 |  | MaskRCNN (swint), MaskRCNN Swin, Swin B - Unet, Swin L - Une |
| vit | 0.08 | 2 | 2.2/27.6 |  | ViT-Adapter-L, maxvit_base, maxvit_tiny_tf_512.in1k |

## medical — loss （KB 覆盖率 26%）

| kb_id | support | breadth | votes/total | passed | raw（归并痕迹） |
|---|---|---|---|---|---|
| bce_dice_loss | 0.44 | 3 | 3.4/7.8 |  | BCE and Dice weighted loss function with a 1:3 loss function |
| focal_loss | 0.38 | 3 | 3.0/7.8 |  | BoundaryLoss + 0.5 Focal Symmetric Loss, FocalLoss, binary-f |
| cross_entropy_loss | 0.18 | 1 | 1.4/7.8 |  | BCE loss + Tversky loss, CE:LovaszLoss=1:3 |

# task_type = object_detection

## class_imbalance — backbone·车架 frame （KB 覆盖率 73%，本组占全部 backbone 票 21%）

| kb_id | support | breadth | votes/total | passed | raw（归并痕迹） |
|---|---|---|---|---|---|
| unet | 0.61 | 2 | 4.3/7.1 | ✔ | 3D U-Net, 3D U-Net, 3D U-Net, 3D U-Net, DynUnet, MONAI’s 3D  |
| yolov8 | 0.39 | 1 | 2.8/7.1 |  | Ultralytics YOLO, Ultralytics YOLO, Ultralytics YOLO, Ultral |

## class_imbalance — backbone·发动机 engine （KB 覆盖率 73%，本组占全部 backbone 票 52%）

| kb_id | support | breadth | votes/total | passed | raw（归并痕迹） |
|---|---|---|---|---|---|
| resnet | 0.45 | 4 | 7.8/17.4 |  | 2.5D UNet with timm's ResNet34d, MONAI’s SegResNet, 3D Resne |
| efficientnet | 0.35 | 4 | 6.1/17.4 |  | 2.5D UNet with timm's EfficientNet-B2, 2.5D UNet with timm's |
| convnext | 0.11 | 2 | 2.0/17.4 |  | 2.5D UNet with timm's ConvNeXt-Nano, 3D ConvNeXt-like model, |
| swin_transformer | 0.09 | 1 | 1.5/17.4 |  | Swin Transformer pretrained on NIH Chest X-rays, Swin Transf |

## class_imbalance — loss （KB 覆盖率 38%）

| kb_id | support | breadth | votes/total | passed | raw（归并痕迹） |
|---|---|---|---|---|---|
| cross_entropy_loss | 0.71 | 4 | 8.1/11.4 | ✔dom(×2.45,+0.421) | BCE loss, BCE{4-class} + [0.75* lovasz_loss + 0.25* BCE ]{Se |
| focal_loss | 0.29 | 2 | 3.3/11.4 |  | FocalTversky++ loss, aux loss = 0.6weighted bce + 0.4dice, w |

## medical — backbone·发动机 engine （KB 覆盖率 61%，本组占全部 backbone 票 61%）

| kb_id | support | breadth | votes/total | passed | raw（归并痕迹） |
|---|---|---|---|---|---|
| efficientnet | 0.61 | 2 | 4.6/7.5 | ✔ | EfficientNet-B6, EfficientNet-B6, EfficientNet-B4, Efficient |
| swin_transformer | 0.20 | 1 | 1.5/7.5 |  | Swin Transformer pretrained on NIH Chest X-rays, Swin Transf |
| resnet | 0.19 | 2 | 1.4/7.5 |  | RetinaNet (ResNet101), RetinaNet (ResNet152), SeResnet152-Un |

## medical — loss （KB 覆盖率 37%）

| kb_id | support | breadth | votes/total | passed | raw（归并痕迹） |
|---|---|---|---|---|---|
| cross_entropy_loss | 0.78 | 2 | 2.9/3.7 | ✔ | BCE loss, BCE{4-class} + [0.75* lovasz_loss + 0.25* BCE ]{Se |
| focal_loss | 0.22 | 1 | 0.8/3.7 |  | aux loss = 0.6weighted bce + 0.4dice, weight of weighted bce |

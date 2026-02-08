# Kết Quả Baseline Từ Bài Báo Gốc - OpenEarthMap

## Dataset Info
- **Tổng**: 5000 ảnh, 97 vùng, 44 quốc gia, 6 lục địa
- **Độ phân giải**: 0.25-0.5m GSD, 1024×1024 pixels
- **Chia dữ liệu**: Train/Val/Test = 3000/500/1500
- **8 Lớp**: Bareland(1.5%), Rangeland(22.9%), Developed(16.1%), Road(6.7%), Tree(20.2%), Water(3.3%), Agriculture(13.7%), Building(15.6%)

## Fully Supervised Segmentation (mIoU %)

### CNN-based
| Model | Backbone | mIoU |
|-------|----------|------|
| U-Net | VGG-11 | 64.97 |
| U-Net | ResNet-34 | 65.43 |
| U-Net | EfficientNet-B4 | **68.20** |
| DeepLabV3 | ResNet-50 | 63.16 |
| HRNet | W48 | 63.07 |

### Transformer-based
| Model | Backbone | mIoU |
|-------|----------|------|
| SegFormer | MiT-B5 | 64.01 |
| SETR PUP | ViT-L | 61.40 |
| UPerNet | ViT | 60.52 |
| UPerNet | Swin-B | **66.09** |
| UPerNet | Twins | 64.05 |
| UPerNet | ConvNeXt | 61.93 |
| K-Net | Swin-B | 66.11 |

### Hybrid
| Model | Backbone | mIoU |
|-------|----------|------|
| U-NetFormer | ResNeXt101 | 68.37 |
| FT-U-NetFormer | Swin-B | **69.13** ⭐ BEST |

### IoU Theo Lớp (FT-U-NetFormer - Swin-B)
| Class | IoU (%) |
|-------|--------|
| Water | 87.44 |
| Building | 80.29 |
| Agriculture | 77.50 |
| Tree | 73.33 |
| Road | 65.85 |
| Rangeland | 60.84 |
| Developed | 57.58 |
| Bareland | 50.19 |

## Limited Labels (10% Training Data = 300 ảnh)
| Model | mIoU |
|-------|------|
| U-Net + EfficientNet-B4 | **60.18** |
| K-Net | 54.33 |
| UPerNet + Swin-B | 52.39 |
| SegFormer | 51.61 |

> **Kết luận**: CNN vượt Transformer 6-15% khi dữ liệu ít

## UDA (SegFormer-based)
| Method | mIoU |
|--------|------|
| Oracle | 64.99 |
| DAFormer | **62.35** |
| Source only | 59.32 |

## NAS (Neural Architecture Search)
| Model | mIoU | Params |
|-------|------|--------|
| SparseMask | 60.21 | 2.96M |
| FasterSeg | 59.41 | 3.47M |

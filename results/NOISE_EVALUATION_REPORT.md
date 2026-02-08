# Danh Gia Cac Kieu Tao Nhieu

## 1. mIoU theo Noise Type va Noise Rate

| Noise Type | 5% | 10% | 15% | 20% | 30% |
|---|---|---|---|---|---|
| random_flip | 64.00±10.0% | 54.39±8.5% | 47.12±7.5% | 41.24±6.6% | 32.03±5.2% |
| boundary | 99.18±0.8% | 98.94±1.3% | 93.80±6.2% | 94.06±8.0% | 95.26±8.2% |
| region_swap | 0.88±0.5% | 1.20±2.5% | 0.40±0.5% | 0.71±0.7% | 0.66±0.6% |
| confusion_based | 64.93±9.9% | 56.35±8.4% | 49.68±7.4% | 44.11±6.5% | 35.02±5.0% |
| mixed | 0.60±0.6% | 0.68±0.5% | 0.69±0.5% | 1.44±2.0% | 1.55±1.7% |

## 2. Dac Diem Tung Loai Nhieu


### random_flip
- Rate 5%: actual=5.00%, boundary_ratio=2.46%
- Rate 10%: actual=10.00%, boundary_ratio=2.42%
- Rate 15%: actual=15.00%, boundary_ratio=2.42%
- Rate 20%: actual=20.00%, boundary_ratio=2.39%
- Rate 30%: actual=30.00%, boundary_ratio=2.43%

### boundary
- Rate 5%: actual=0.18%, boundary_ratio=100.00%
- Rate 10%: actual=0.26%, boundary_ratio=95.00%
- Rate 15%: actual=1.07%, boundary_ratio=58.67%
- Rate 20%: actual=1.24%, boundary_ratio=44.36%
- Rate 30%: actual=1.16%, boundary_ratio=51.47%

### region_swap
- Rate 5%: actual=94.49%, boundary_ratio=2.28%
- Rate 10%: actual=92.46%, boundary_ratio=2.44%
- Rate 15%: actual=97.37%, boundary_ratio=2.38%
- Rate 20%: actual=95.53%, boundary_ratio=2.34%
- Rate 30%: actual=95.84%, boundary_ratio=2.32%

### confusion_based
- Rate 5%: actual=5.00%, boundary_ratio=2.38%
- Rate 10%: actual=10.00%, boundary_ratio=2.43%
- Rate 15%: actual=15.00%, boundary_ratio=2.38%
- Rate 20%: actual=20.00%, boundary_ratio=2.39%
- Rate 30%: actual=30.00%, boundary_ratio=2.43%

### mixed
- Rate 5%: actual=96.06%, boundary_ratio=2.32%
- Rate 10%: actual=96.15%, boundary_ratio=2.31%
- Rate 15%: actual=96.68%, boundary_ratio=2.36%
- Rate 20%: actual=91.38%, boundary_ratio=2.46%
- Rate 30%: actual=91.76%, boundary_ratio=2.41%

## 3. Anh Huong Theo Class (noise_rate=15%)

| Class | random_flip | boundary | region_swap | confusion_based | mixed |
|---|---|---|---|---|---|
| Bareland | 37.9% | 97.2% | 0.0% | 39.6% | 0.5% |
| Rangeland | 48.7% | 95.9% | 0.5% | 40.5% | 0.8% |
| Developed | 45.2% | 93.8% | 1.1% | 54.4% | 1.4% |
| Road | 48.1% | 81.8% | 0.0% | 46.9% | 0.9% |
| Tree | 54.4% | 95.0% | 0.1% | 61.7% | 0.4% |
| Water | 42.5% | 96.2% | 1.1% | 50.1% | 0.6% |
| Agriculture | 59.6% | 97.6% | 0.2% | 59.2% | 0.5% |
| Building | 40.6% | 98.6% | 0.3% | 45.1% | 0.6% |

## 4. Phan Tich & Nhan Xet

### Dac diem chinh cua tung loai nhieu:

| Loai | Dac diem | Ung dung mo phong |
|------|---------|-------------------|
| **Random Flip** | Dom dom, khong co cau truc, anh huong deu | Model du doan sai ngau nhien |
| **Boundary** | Tap trung o bien, co cau truc spatial | Model khong xac dinh duoc ranh gioi |
| **Region Swap** | Anh huong ca vung lon, it pixels nhung impact lon | Model nham toan bo mot khu vuc |
| **Confusion-based** | Theo phan phoi loi thuc te, realistic nhat | Pseudo-label tu model thuc |
| **Mixed** | Ket hop tat ca, da dang nhat | Tinh huong thuc te phuc tap |

### Class de bi anh huong nhat:
- **Bareland** (50.19% IoU baseline) - Dien tich nho, de nham voi Developed
- **Developed** (57.58% IoU baseline) - Ranh gioi mo voi Bareland va Road
- **Rangeland** (60.84% IoU baseline) - De nham voi Agriculture

### Khuyen nghi cho DAE:
- Train DAE voi **Mixed noise** de robust voi nhieu tinh huong
- Chu y **Boundary noise** vi la loi pho bien nhat trong segmentation
- DAE can hoc duoc **confusion patterns** giua cac cap class de nham
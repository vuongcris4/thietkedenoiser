"""
Quản lí cấu hình thí nghiệm
Khi chạy lệnh train -> ĐẦU TIÊN được gọi
1. Đọc file YAML
2. Xử lý kế thừa config cha, áp dụng ghi đè từ CLI
3. Trả về 1 dict Python chứa toàn bộ hyperparameters.

CLI (python train.py --config ...) 
→ load_config_from_args () 
    → load_config() 
        → load_yaml()                   # đọc file
        → load_config() [base.yaml]     # đệ quy nếu có _base_
        → _deep_merge ()                # gộp config cha + con (overwrite con vào cha)
                base = {"a": {"x": 1, "y": 2}}, override = {"a": {"y": 9}}
                → kết quả: {"a": {"x": 1, "y": 9}}  (y bị ghi đè)
        → _set_nested() 
                Ví dụ: _set_nested(cfg, "training.lr", 0.001)
                → cfg["training"]["lr"] = 0.001
        → print_config()  
"""
import os
import sys
import copy
import argparse
import yaml
from typing import Any, Dict, Optional


def _deep_merge(base: dict, override: dict) -> dict:
    """Gộp sâu (recursive merge) dict `override` vào dict `base`.

    - Duyệt từng key trong `override`:
      + Nếu key tồn tại ở cả 2 dict VÀ cả 2 giá trị đều là dict → gộp đệ quy (đi sâu vào).
      + Ngược lại → giá trị từ `override` sẽ GHI ĐÈ giá trị trong `base`.
    - Trả về dict mới, không thay đổi dict gốc (dùng deepcopy).

    Ví dụ: base = {"a": {"x": 1, "y": 2}}, override = {"a": {"y": 9}}
           → kết quả: {"a": {"x": 1, "y": 9}}  (y bị ghi đè)
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _set_nested(d: dict, key_path: str, value: Any) -> None:
    """Gán giá trị vào dict lồng nhau thông qua đường dẫn key dùng dấu chấm.

    - Tách key_path theo dấu "." thành danh sách các key.
    - Đi sâu vào dict theo từng key, tạo dict rỗng nếu chưa tồn tại.
    - Gán giá trị cuối cùng vào key cuối.

    Ví dụ: _set_nested(cfg, "training.lr", 0.001)
           → cfg["training"]["lr"] = 0.001
    """
    keys = key_path.split(".")
    for k in keys[:-1]:
        if k not in d or not isinstance(d[k], dict):
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value

# str -> python data type
def _parse_value(value_str: str) -> Any:
    """Chuyển đổi chuỗi ký tự thành kiểu dữ liệu Python phù hợp.

    Thứ tự thử chuyển đổi:
      1. None/null/~ → None
      2. true/yes → True, false/no → False
      3. Số nguyên (int)
      4. Số thực (float)
      5. Danh sách (nếu có dấu phẩy → tách và parse từng phần tử)
      6. Giữ nguyên chuỗi nếu không khớp gì

    Hàm này được dùng khi parse các giá trị ghi đè từ CLI, ví dụ:
      --override training.lr=0.001  → float 0.001
      --override training.epochs=100 → int 100
    """
    # None
    if value_str.lower() in ("none", "null", "~"):
        return None
    # Bool
    if value_str.lower() in ("true", "yes"):
        return True
    if value_str.lower() in ("false", "no"):
        return False
    # Int
    try:
        return int(value_str)
    except ValueError:
        pass
    # Float
    try:
        return float(value_str)
    except ValueError:
        pass
    # List (comma separated)
    if "," in value_str:
        return [_parse_value(v.strip()) for v in value_str.split(",")]
    # String
    return value_str


# YAML -> Dict Python
def load_yaml(path: str) -> dict:
    """Đọc và parse một file YAML thành dict Python.

    Trả về dict rỗng {} nếu file YAML trống (None).
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}

# load config từ YAML, hỗ trợ kế thừa và ghi đè
#     Ví dụ: file dae_resnet34.yaml có dòng `_base_: default.yaml`
#           → default.yaml được load trước, rồi dae_resnet34.yaml ghi đè lên.

def load_config(config_path: str, overrides: Optional[Dict[str, Any]] = None) -> dict:
    """Hàm chính để load cấu hình từ file YAML, có hỗ trợ kế thừa và ghi đè.

    Luồng hoạt động:
      1. Đọc file YAML tại config_path.
      2. Nếu trong file có trường "_base_": "tên_file_cha.yaml":
         → Load config cha (đệ quy, hỗ trợ chuỗi kế thừa nhiều cấp).
         → Gộp config cha + config con (con ghi đè cha).
      3. Áp dụng các giá trị ghi đè (overrides) nếu có.
      4. Trả về dict config đã merge hoàn chỉnh.

    Ví dụ: file dae_resnet34.yaml có dòng `_base_: default.yaml`
           → default.yaml được load trước, rồi dae_resnet34.yaml ghi đè lên.

    Args:
        config_path: Đường dẫn tới file YAML.
        overrides: Dict ghi đè, key dùng dấu chấm, vd {"training.lr": 0.001}.

    Returns:
        Dict cấu hình đã merge.
    """
    cfg = load_yaml(config_path)
    config_dir = os.path.dirname(os.path.abspath(config_path))

    # Handle _base_ inheritance
    if "_base_" in cfg:
        base_path = cfg.pop("_base_")
        if not os.path.isabs(base_path):
            base_path = os.path.join(config_dir, base_path)
        base_cfg = load_config(base_path)  # Recursive for chained bases
        cfg = _deep_merge(base_cfg, cfg)

    # Apply overrides
    if overrides:
        for key_path, value in overrides.items():
            _set_nested(cfg, key_path, value)

    return cfg


# Hàm entry point
def load_config_from_args(args: Optional[argparse.Namespace] = None) -> dict:
    """Load config từ tham số dòng lệnh (CLI) — dùng khi chạy train từ terminal.

    Luồng hoạt động:
      1. Parse các tham số CLI:
         --config <path>       (bắt buộc) Đường dẫn tới file YAML
         --override key=value  (tuỳ chọn, lặp nhiều lần) Ghi đè giá trị config
         --resume <path>       (tuỳ chọn) Đường dẫn checkpoint để resume training
      2. Chuyển các chuỗi override thành dict {key: value} (dùng _parse_value).
      3. Gọi load_config() để load + merge config.
      4. Nếu có --resume → thêm trường "resume" vào config.

    Ví dụ chạy:
      python train.py --config configs/dae_resnet34.yaml --override training.lr=0.001

    Returns:
        Dict cấu hình hoàn chỉnh, đã áp dụng mọi giá trị ghi đè từ CLI.
    """
    if args is None:
        parser = argparse.ArgumentParser(description="Train with YAML config")
        parser.add_argument("--config", type=str, required=True,
                            help="Path to YAML config file")
        parser.add_argument("--override", type=str, nargs="*", default=[],
                            help="Override config values: key=value (dot notation)")
        parser.add_argument("--resume", type=str, default=None,
                            help="Path to checkpoint to resume from")
        args = parser.parse_args()

    # Parse overrides
    overrides = {}
    override_list = getattr(args, "override", []) or []
    for ov in override_list:
        if "=" not in ov:
            print(f"Warning: ignoring malformed override '{ov}' (expected key=value)")
            continue
        key, val = ov.split("=", 1)
        overrides[key] = _parse_value(val)

    cfg = load_config(args.config, overrides)

    # Inject resume path
    if getattr(args, "resume", None):
        cfg["resume"] = args.resume

    return cfg


def cfg_to_flat(cfg: dict, prefix: str = "") -> dict:
    """Làm phẳng dict config lồng nhau thành các key dùng dấu chấm.

    Dùng để in log hoặc gửi lên wandb/tensorboard.

    Ví dụ: {"training": {"lr": 0.001, "epochs": 100}}
           → {"training.lr": 0.001, "training.epochs": 100}
    """
    flat = {}
    for k, v in cfg.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(cfg_to_flat(v, full_key))
        else:
            flat[full_key] = v
    return flat


def print_config(cfg: dict, title: str = "Configuration") -> None:
    """In đẹp toàn bộ config ra terminal để dễ kiểm tra.

    Gọi cfg_to_flat() để làm phẳng, rồi in từng key-value theo thứ tự ABC.
    Thường được gọi ở đầu quá trình train để xác nhận config đúng.
    """
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    flat = cfg_to_flat(cfg)
    max_key_len = max(len(k) for k in flat.keys()) if flat else 0
    for k, v in sorted(flat.items()):
        print(f"  {k:<{max_key_len}}  = {v}")
    print(f"{'=' * 60}\n")


# MỤC ĐÍCH: load YAML và return 1 Dict Python
# Hỗ trợ: kế thừa (_base_) + ghi đè từ CLI (override) + In debug
if __name__ == "__main__":
    # Quick test
    if len(sys.argv) > 1:
        cfg = load_config_from_args()
        print_config(cfg)
    else:
        print("Usage: python config.py --config configs/dae_resnet34.yaml")
        print("       python config.py --config configs/diffusion.yaml --override training.lr=0.001")

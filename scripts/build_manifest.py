# scripts/build_manifest.py
from pathlib import Path
from tqdm import tqdm
import json
import pandas as pd

# ===== [사용자 지정: 프로젝트 루트 경로] =====
# 슬래시(/) 사용 또는 문자열 앞에 r을 붙여 raw 문자열로 작성
PROJECT_ROOT = Path(r"C:/Users/tim12/Desktop/deepfakeAI_project")

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR = PROCESSED_DIR
OUT_CSV = OUT_DIR / "manifest_v1.csv"
OUT_JSONL = OUT_DIR / "manifest_v1.jsonl" # jsonl도 함께 작성

# 이미지 확장자
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 라벨 매핑
LABEL_MAP = {"real": 0, "fake": 1}

def is_image(p: Path) -> bool:
    return (p.is_file() and p.suffix.lower() in IMG_EXTS)

def rows_from_simple_tree(base: Path, source_name: str, eval_subset: str):
    """
    base/real/*.jpg, base/fake/*.jpg 같은 '간단 2클래스' 구조를 통째로 긁어온다.
    (하위에 폴더가 더 있어도 rglob로 이미지만 모음)
    """
    rows = []
    for cls in ["real", "fake"]:
        cls_dir = base / cls
        if not cls_dir.exists():
            continue
        for img in tqdm(sorted(cls_dir.rglob("*")), desc=f"[scan] {source_name}/{cls}", leave=False):
            if is_image(img):
                label = LABEL_MAP[cls]
                # 이미지 데이터는 '파일명(stem)' 기준으로 개별 그룹 부여
                group_id = f"{source_name}::{img.stem}"
                # 프로젝트 루트 기준 상대 경로
                rel_path = img.resolve().relative_to(PROJECT_ROOT)
                rows.append({
                    "path": str(rel_path).replace("\\", "/"),
                    "label": label,
                    "group_id": group_id,
                    "source": source_name,
                    "eval_subset": eval_subset,
                })
    return rows

def rows_from_frames(frames_root: Path, source_name = "frames:DFD", eval_subset="unsplit"):
    """
    frames/real/<video_id>/*.jpg, frames/fake/<video_id>/*.jpg 구조 처리.
    영상 폴더(<video_id>) 단위로 group_id를 부여한다.
    """
    rows = []
    for cls in ["real", "fake"]:
        cls_dir = frames_root / cls
        if not cls_dir.exists():
            continue
        # 1차 하위 폴더: video_id
        for video_dir in tqdm(sorted([d for d in cls_dir.iterdir() if d.is_dir()]),
                              desc=f"[scan] {source_name}/{cls}", leave=False):
            video_id = video_dir.name
            group_id = f"{source_name}::{video_id}"
            for img in sorted(video_dir.rglob("*")):
                if is_image(img):
                    label = LABEL_MAP[cls]
                    rel_path = img.resolve().relative_to(PROJECT_ROOT)
                    rows.append({
                        "path": str(rel_path).replace("\\", "/"),
                        "label": label,
                        "group_id": group_id,
                        "source": source_name,
                        "eval_subset": eval_subset
                    })
    return rows

def build_manifest():
    all_rows = []

    # 1) Real_Fake -> 아직 분할 전이라 eval_subset="unsplit"
    rf_root = RAW_DIR / "Real_Fake"
    if rf_root.exists():
        all_rows += rows_from_simple_tree(rf_root, "Real_Fake", "unsplit")

    # 2) Deepfake_Real -> Train/Validation/Test 각각 기록
    dfr_root = RAW_DIR / "Deepfake_Real"
    if dfr_root.exists():
        for subfolder, subset_name in [("Train", "train"), ("Validation", "val"), ("Test", "test")]:
            base = dfr_root / subfolder
            if base.exists():
                all_rows += rows_from_simple_tree(base, f"Deepfake_Real/{subfolder}", subset_name)

    # 3) frames (DFD에서 추출한 프레임) -> 아직 분할 전이라 eval_subset="unsplit"
    frames_root = PROCESSED_DIR / "frames"
    if frames_root.exists():
        all_rows += rows_from_frames(frames_root, "frames:DFD", "unsplit")

    # DataFrame 생성
    df = pd.DataFrame(all_rows)
    if df.empty:
        raise SystemExit("⚠️ 수집된 이미지가 없습니다. 경로/폴더 구조를 확인하세요.")

    # 동일 path 중복 제거 + 보기 좋게 정렬
    df = (
        df.drop_duplicates(subset=["path","source"])
            .sort_values(by=["source", "group_id", "path"])
            .reset_index(drop=True)
    )

    # 저장
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for rec in df.to_dict(orient="records"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 요약 출력
    print(f"✅Saved: {OUT_CSV} (rows={len(df)})")
    print(f"✅Saved: {OUT_JSONL} (rows={len(df)})")
    print("\n[Quick report]")
    print("label counts:\n", df["label"].value_counts().rename({0: "REAL", 1: "FAKE"}))
    print("\nby source:\n", df["source"].value_counts())
    print("\nby eval_subset:\n", df["eval_subset"].value_counts())

if __name__ == "__main__":
    build_manifest()
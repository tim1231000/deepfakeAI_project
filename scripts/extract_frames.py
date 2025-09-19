import argparse # 명령행 인자 파싱용
import cv2 # OpenCV: 비디오/이미지 I/O
import numpy as np # 수치 연산
from pathlib import Path # 경로 다루기 편하게
from tqdm import tqdm # 진행률 표시


def list_videos(src_dir: Path):
    """
    주어진 src_dir(data/raw/DFD_videos) 아래의 real/, fake/ 폴더를 찾아 
    각 클래스별 비디오 파일 경로 리스트를 반환한다.
    """
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
    class_dirs = []

    for cls in ["real", "fake"]:
        cdir = src_dir / cls
        if not cdir.exists():
            print(f"[경고] {cdir} 폴더가 없음. 건너뜀.")
            continue
        class_dirs.append(cdir)

    videos = []
    for cdir in class_dirs:
        for p in cdir.rglob("*"):
            if p.suffix.lower() in video_exts:
                # (비디오경로, 클래스명) 튜플로 저장
                videos.append((p, cdir.name))
    
    return videos # [(Path, "real"/"fake"), ...]


def get_video_info(cap: cv2.VideoCapture):
    """
    VideoCapture로부터 FPS, 프레임 수, 길이(초)를 얻는다.
    일부 코덱/파일에서는 FPS나 프레임 수가 0이 나올 수 있다.
    그러면 길이 추정이 부정확할 수 있다.
    """
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    # FPS 또는 프레임 수가 0이면 길이 계산이 어려움 -> 나중 로직에서 보정
    if fps is None or fps <= 0:
        fps = 0.0
    if total_frames is None or total_frames <= 0:
        total_frames = 0
    
    duration = (total_frames / fps) if (fps > 0 and total_frames > 0) else 0.0

    return fps, int(total_frames), float(duration)


def make_sample_times(duration_sec: float, target_fps: float, min_frames_if_short: int, max_frames_per_video: int):
    """
    길이(초)와 타겟 1fps를 기준으로 샘플링 시간들을(sec) 만든다.
    - 기본: 0, 1, 2, ..., float(duration)
    - 3초 미만: 최소 min_frames_if_short(기본 3)장을 균등 간격으로 뽑음
    - 너무 많으면(> max_frames_per_video): 균등하게 max개로 다운샘플
    """
    
    # 안정장치: duration이 0으로 잡힌 영상 대비
    if duration_sec <= 0:
        # 길이를 알 수 없다면 일단 3장을 가정 (0s, 0.5s, 1.0s 같은 느낌)
        # 실제 추출 시 프레임 인덱스 변환 단계에서 보정됨
        raw_times = np.linspace(0, 2, num=min_frames_if_short, endpoint=True)
    else:
        if duration_sec < 3:
            # 3초 미만 -> 최소 N장 균등 분할
            raw_times = np.linspace(0, duration_sec, num=min_frames_if_short, endpoint=True)
        else:
            # 1fps 규칙: 0~duration까지 1초 간격
            # 예: duration 10.7s -> 0, 1, 2, ..., 10 (총 11개)
            last = int(np.floor(duration_sec))
            raw_times = np.arange(0, last + 1, 1.0)

    # 상한 (최대 프레임 수) 적용: 많으면 균등하게 max개로 줄이기
    if len(raw_times) > max_frames_per_video:
        raw_times = np.linspace(raw_times[0], raw_times[-1], num=max_frames_per_video, endpoint=True)

    return raw_times.astype(float)


def times_to_frame_indices(times_sec: np.ndarray, fps_src: float, total_frames: int):
    """
    샘플링 시간(sec) 배열을 원본 비디오의 프레임 인덱스로 변환한다.
    - 인덱스 = round(time * fps_src)
    - 영상 끝(=total_frames-1)을 넘지 않도록 클리핑
    - 중복 인덱스 제거(안정성)
    """

    if fps_src <= 0 or total_frames <= 0:
        # FPS/프레임 수 정보를 믿을 수 없으면, 일단 처음 3프레임 가정
        idx = np.arange(min(3, max(1, total_frames)))
        return np.unique(idx)
    
    idx = np.rint(times_sec * fps_src).astype(int)
    idx = np.clip(idx, 0, max(0, total_frames - 1))
    # 정렬 + 중복 제거
    idx = np.unique(idx)
    return idx

def extract_frames_from_video(
        video_path: Path,
        out_root: Path,
        cls_name: str,
        target_fps: float = 1.0,
        min_frames_if_short: int = 3,
        max_frames_per_video: int = 100,
        jpeg_quality: int = 90,
):
    """
    단일 비디오에서 규칙에 맞춰 프레임을 추출하여 저장한다.
    저장 경로: out_root/cls_name/<비디오이름(base) without ext>/frame_xxxx.jpg
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[경고] 열 수 없는 비디오: {video_path}")
        return 0
    
    fps_src, total_frames, duration = get_video_info(cap)

    # 1) 샘플링할 '시간(sec)' 배열 생성
    time_sec = make_sample_times(duration, target_fps, min_frames_if_short, max_frames_per_video)
    # 2) 해당 시간을 원본 프레임 인덱스로 변환
    target_indices = times_to_frame_indices(time_sec, fps_src, total_frames)

    # 출력 폴더 준비
    vid_stem = video_path.stem # 확장자 없는 파일명
    out_dir = out_root / cls_name / vid_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # 프레임 순환하며 원하는 인덱스만 저장 (순차 읽기가 안전)
    saved = 0
    current_idx = 0
    target_set = set(target_indices.tolist())

    # 일부 코덱에서 임의 seek는 부정확할 수 있어 '순차 읽기 + 선택 저장'이 호환성이 좋다.
    with tqdm(total=total_frames if total_frames > 0 else None, leave=False, desc=f"{cls_name}/{vid_stem}") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break # 끝
            
            # 저장 대상이면 저장
            if current_idx in target_set:
                out_path = out_dir / f"frame_{saved+1:04d}.jpg"
                cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
                saved += 1
                # 다 저장했으면 조기 종료
                if saved >= len(target_indices):
                    # 남은 프레임은 더 읽지 않고 종료
                    break
                
            current_idx += 1
            if pbar.total is not None:
                pbar.update(1)

    cap.release()

    # 예외: 아주 짧거나 메타가 이상해서 하나도 못 뽑힌 경우
    if saved == 0:
        # 가능한 만큼 처음 몇 프레임이라도 시도
        cap = cv2.VideoCapture(str(video_path))
        fallback_needed = min(3, max_frames_per_video)
        i = 0
        while i < fallback_needed:
            ret, frame = cap.read()
            if not ret:
                break
            out_path = out_dir / f"frame_{i+1:04d}.jpg"
            cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
            i += 1

        cap.release()
        saved = i

    return saved
    

def main():
    parser = argparse.ArgumentParser(description="Extract frames at 1fps with min-3-if-short & max-100 rules.")
    parser.add_argument("--src_dir", type=str, required=True, help="원본 비디오 루트 (예: data/raw/DFd_videos)")
    parser.add_argument("--out_dir", type=str, required=True, help="프레임 출력 루트 (예: data/processed/frames_1fps_min3_max100)")
    parser.add_argument("--fps", type=float, default=1.0, help="초당 수출 프레임 수(기본 1.0)")
    parser.add_argument("--min_frames_if_short", type=int, default=3, help="3초 미만일 때 최소 프레임 수(기본 3)")
    parser.add_argument("--max_frames", type=int, default=100, help="영상당 최대 프레임 수(기본 100)")
    parser.add_argument("--jpeg_quality", type=int, default=95, help="저장 JPEG 품질(기본 95)")
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = list_videos(src_dir)
    print(f"[정보] 찾은 비디오 수: {len(videos)}개")
    
    total_saved = 0
    for video_path, cls_name in tqdm(videos, desc="전체 진행"):
        saved = extract_frames_from_video(
            video_path = video_path,
            out_root = out_dir,
            cls_name = cls_name,
            target_fps = args.fps,
            min_frames_if_short = args.min_frames_if_short,
            max_frames_per_video = args.max_frames,
            jpeg_quality = args.jpeg_quality,
        )
        total_saved += saved

    print(f"[완료] 저장한 전체 프레임: {total_saved}장")

if __name__ == "__main__":
    main()
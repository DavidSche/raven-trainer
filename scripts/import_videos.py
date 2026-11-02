"""Import and organize D6 video files by camera into storage directory."""
import shutil
import re
from pathlib import Path
from collections import defaultdict

# Source and destination
video_source = Path(r"D:\D6-45\downloaded_videos")
storage_base = Path(r"E:\study-place\Davidche\2026\raven\raven-trainer\data\storage\videos\d6_workshop")

# Patterns for two naming formats
# Format A: {ZONE}-{SUBZONE}-{CAMERA}-{YYYYMMDDHHMM}.mp4 (e.g., 04-01-01-202606261500.mp4)
# Format B: {ZONE}-{SUBZONE}-{CAMERA}-{UNIX_MS_TIMESTAMP}.mp4 (e.g., 4-1-1-0000000952200000000.mp4)
# Also: 07-m01-01_0601_0106-0217.mp4, 07-M04-01-20260522-10-11.mp4

pattern_a = re.compile(r'^(\d{2})-(\d{2})-(\d{2})-(\d{12,})\.mp4$', re.IGNORECASE)
pattern_b = re.compile(r'^(\d)-(\d)-(\d)-(\d+)\.mp4$', re.IGNORECASE)
pattern_07 = re.compile(r'^07-(m\d{2})-(\d{2}).*\.mp4$', re.IGNORECASE)

camera_map = defaultdict(list)
unmatched = []

for video in sorted(video_source.glob("*.mp4")):
    name = video.name.lower()
    matched = False

    # Try Format A (e.g., 04-01-01-202606261500.mp4)
    m = pattern_a.match(name)
    if m:
        zone, subzone, camera = m.group(1), m.group(2), m.group(3)
        cam_key = f"{zone}-{subzone}-{camera}"
        camera_map[cam_key].append(video)
        matched = True

    # Try Format B (e.g., 4-1-1-0000000952200000000.mp4)
    if not matched:
        m = pattern_b.match(name)
        if m:
            zone, subzone, camera = m.group(1), m.group(2), m.group(3)
            # Pad with leading zeros for consistency
            cam_key = f"{int(zone):02d}-{int(subzone):02d}-{int(camera):02d}"
            camera_map[cam_key].append(video)
            matched = True

    # Try 07 series (e.g., 07-m01-01_0601_0106-0217.mp4, 07-M04-01-20260522-10-11.mp4)
    if not matched:
        m = pattern_07.match(name)
        if m:
            subzone, camera = m.group(1).lower(), m.group(2)
            cam_key = f"07-{subzone}-{camera}"
            camera_map[cam_key].append(video)
            matched = True

    if not matched:
        unmatched.append(video)

# Print summary
print("=== Video Organization Summary ===")
print(f"Total videos: {len(list(video_source.glob('*.mp4')))}")
print(f"Matched: {sum(len(v) for v in camera_map.values())}")
print(f"Unmatched: {len(unmatched)}")
print()

print("=== Camera Distribution ===")
for cam in sorted(camera_map.keys()):
    print(f"  {cam}: {len(camera_map[cam])} videos")

if unmatched:
    print("\n=== Unmatched videos ===")
    for v in unmatched:
        print(f"  {v.name}")

# Copy videos to organized storage directories
print("\n=== Copying videos to storage ===")
total_copied = 0
for cam_key, videos in sorted(camera_map.items()):
    cam_dir = storage_base / cam_key
    cam_dir.mkdir(parents=True, exist_ok=True)
    for video in videos:
        dest = cam_dir / video.name
        if not dest.exists():
            shutil.copy2(video, dest)
            total_copied += 1
    print(f"  {cam_key}: copied {len(videos)} videos")

print(f"\nTotal copied: {total_copied}")

# Verify
print("\n=== Storage verification ===")
for cam_dir in sorted(storage_base.iterdir()):
    if cam_dir.is_dir():
        count = len(list(cam_dir.glob("*.mp4")))
        print(f"  {cam_dir.name}: {count} videos")

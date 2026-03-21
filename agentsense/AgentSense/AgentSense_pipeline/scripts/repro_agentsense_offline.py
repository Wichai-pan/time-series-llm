#!/usr/bin/env python3
import os
import re
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "offline_repro"
OUT.mkdir(parents=True, exist_ok=True)

STEP4_OUT = OUT / "step_4_data"
STEP5_OUT = OUT / "step_5_data"
STEP9_OUT = OUT / "step_9_data"


def split_weekly_routine_by_day(input_file_path: Path, output_dir: Path):
    days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = input_file_path.read_text(encoding="utf-8").splitlines(keepends=True)
    base_name = input_file_path.stem

    current_day = None
    buffer = []

    for line in lines:
        day_match = next((day for day in days_of_week if line.strip().startswith(f"{day}:")), None)
        if day_match:
            if current_day and buffer:
                out = output_dir / f"{base_name}_{current_day}.txt"
                out.write_text("".join(buffer), encoding="utf-8")
            current_day = day_match
            buffer = [line]
        else:
            buffer.append(line)

    if current_day and buffer:
        out = output_dir / f"{base_name}_{current_day}.txt"
        out.write_text("".join(buffer), encoding="utf-8")


def parse_routine_file(routine_path: Path):
    routine_content = routine_path.read_text(encoding="utf-8")
    routines = []
    blocks = []
    current_location = None

    for raw in routine_content.splitlines():
        line = raw.strip()
        if not line:
            continue

        location_match = re.match(r"^(\d{1,2}:\d{1,2}) - \d{1,2}:\d{1,2}, (.+?): .+", line)
        if location_match:
            if blocks:
                routines.append(blocks)
                blocks = []
            start_time = location_match.group(1).strip().lower().replace(" ", "")
            current_location = location_match.group(2).strip().lower().replace(" ", "")
            blocks.append(f"[walk] <{current_location}> ({start_time} - {start_time}) ({current_location})")
            continue

        step_match = re.match(r"^Step\s+(\d+):\s*(.+)", line)
        if step_match:
            step_desc = step_match.group(2)
            blocks.append(f"{step_desc} ({current_location})")

    if blocks:
        routines.append(blocks)
    return routines


def clean_and_crop_daily_routine_file(input_file_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_file_path.stem}_parsed.txt"
    parsed_routines = parse_routine_file(input_file_path)
    with output_path.open("w", encoding="utf-8") as f:
        for block in parsed_routines:
            for step in block:
                f.write(step + "\n")
            f.write("\n")


def extract_room_name_from_block(block: str):
    room_matches = re.findall(r"\(([^()]+)\)", block)
    for match in reversed(room_matches):
        if match.lower() in ["bedroom", "kitchen", "bathroom", "livingroom"]:
            return match.lower()
    return "unknown"


def split_blocks_and_save(input_file_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = input_file_path.stem
    content = input_file_path.read_text(encoding="utf-8")
    blocks = [block.strip() for block in content.strip().split("\n\n") if block.strip()]

    previous_room = "bedroom"
    for part_index, i in enumerate(range(0, len(blocks), 4), start=1):
        chunk = blocks[i:i + 4]
        current_room = previous_room
        if len(chunk) == 4:
            previous_room = extract_room_name_from_block(chunk[3])
        output_filename = f"part_{part_index}_{base_name}_{current_room}.txt"
        (output_dir / output_filename).write_text("\n\n".join(chunk), encoding="utf-8")


ACTION_FORMATS = {
    "walk": 1, "run": 1, "walktowards": 1, "walkforward": 0,
    "turnleft": 0, "turnright": 0, "sit": 1, "standup": 0,
    "grab": 1, "open": 1, "close": 1, "put": 2, "putin": 2,
    "switchon": 1, "switchoff": 1, "drink": 1, "touch": 1, "lookat": 1,
}


def load_environment_data(json_file: Path, env_number: int):
    data = json.loads(json_file.read_text(encoding="utf-8"))
    return data[f"env_{env_number}"]


def format_check(post_file: Path, env_data: dict):
    post_lines = post_file.read_text(encoding="utf-8", errors="replace").splitlines()
    total_lines = invalid_lines = missing_actions = unknown_actions = missing_objects = 0
    total_wrong_lines = incorrect_objects = total_checked_objects = 0

    for line in post_lines:
        line = line.strip()
        if not line:
            continue
        total_lines += 1
        mistake_count = 0

        if line == "[]":
            missing_actions += 1
            total_wrong_lines += 1
            continue

        if "<>" in line:
            missing_objects += 1
            mistake_count += 1

        room_match = re.search(r"\(([^()]*)\)\s*$", line)
        room = room_match.group(1).lower() if room_match else "unknown"

        objects = re.findall(r"<(.*?)>", line)
        objects_clean = [obj for obj in objects if obj != ""]

        valid_rooms = {"bathroom", "bedroom", "kitchen", "livingroom"}
        for obj in objects_clean:
            total_checked_objects += 1
            if obj in valid_rooms:
                continue
            if obj not in env_data.get(room, []):
                incorrect_objects += 1
                mistake_count += 1

        action_match = re.match(r"\[(.*?)\]", line)
        action = action_match.group(1) if action_match else None

        if action not in ACTION_FORMATS:
            unknown_actions += 1
            total_wrong_lines += 1
            continue

        expected_objects = ACTION_FORMATS[action]
        if len(objects) != expected_objects:
            invalid_lines += 1
            mistake_count += 1

        if mistake_count > 0:
            total_wrong_lines += 1

    return {
        "file": str(post_file),
        "total_lines": total_lines,
        "total_wrong_lines": total_wrong_lines,
        "total_wrong_line_percentage": (total_wrong_lines / total_lines) * 100 if total_lines else 0,
        "invalid_format_lines": invalid_lines,
        "invalid_format_percentage": (invalid_lines / total_lines) * 100 if total_lines else 0,
        "missing_actions": missing_actions,
        "missing_action_percentage": (missing_actions / total_lines) * 100 if total_lines else 0,
        "unknown_actions": unknown_actions,
        "unknown_action_percentage": (unknown_actions / total_lines) * 100 if total_lines else 0,
        "lines_with_missing_objects": missing_objects,
        "missing_object_percentage": (missing_objects / total_lines) * 100 if total_lines else 0,
        "total_objects_checked": total_checked_objects,
        "incorrect_objects": incorrect_objects,
        "incorrect_object_percentage": (incorrect_objects / total_checked_objects) * 100 if total_checked_objects else 0,
    }


def run_evaluation(post_processed_folder: Path):
    json_file = ROOT / "room_objects.json"
    rows = []
    for filename in sorted(os.listdir(post_processed_folder)):
        if not filename.endswith(".txt"):
            continue
        env_match = re.search(r"env_(\d+)_", filename)
        if not env_match:
            continue
        env_number = int(env_match.group(1))
        env_data = load_environment_data(json_file, env_number)
        rows.append(format_check(post_processed_folder / filename, env_data))
    return rows


if __name__ == "__main__":
    weekly = DATA / "step_3_routine_data" / "Sarah_routine_env_0.txt"
    if not weekly.exists():
        raise FileNotFoundError(f"Missing input: {weekly}")

    # Step 4
    split_weekly_routine_by_day(weekly, STEP4_OUT)

    # Step 5 (batch over generated day files)
    for day_file in sorted(STEP4_OUT.glob("*.txt")):
        clean_and_crop_daily_routine_file(day_file, STEP5_OUT)

    # Step 9 using provided step_8 sample
    step8 = DATA / "step_8_data" / "label_regenerated_Sarah_routine_env_0_Monday_parsed_grounded.txt"
    if not step8.exists():
        raise FileNotFoundError(f"Missing input: {step8}")
    split_blocks_and_save(step8, STEP9_OUT)

    # Evaluation on provided step_7 outputs
    eval_rows = run_evaluation(DATA / "step_7_data")
    eval_path = OUT / "evaluation_step7_summary.json"
    eval_path.write_text(json.dumps(eval_rows, indent=2), encoding="utf-8")

    manifest = {
        "step4_files": len(list(STEP4_OUT.glob("*.txt"))),
        "step5_files": len(list(STEP5_OUT.glob("*.txt"))),
        "step9_files": len(list(STEP9_OUT.glob("*.txt"))),
        "evaluation_files": len(eval_rows),
        "outputs_root": str(OUT),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("[AgentSense] offline reproduction completed")
    print(json.dumps(manifest, indent=2))

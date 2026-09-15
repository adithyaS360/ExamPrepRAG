from __future__ import annotations

import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()


BUCKET = "exam-prep-documents"
RAW_DIR = Path("data/raw")

FOLDERS = [
    "CNS",
    "BDA",
    "PC",
    "IOT",
    "syllabus",
]


def get_client():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SECRET_KEY")

    if not url:
        raise RuntimeError("SUPABASE_URL is not set.")

    if not key:
        raise RuntimeError("SUPABASE_SECRET_KEY is not set.")

    return create_client(url, key)


def download_folder(storage, folder: str) -> int:
    local_dir = RAW_DIR / folder
    local_dir.mkdir(parents=True, exist_ok=True)

    files = storage.list(
        folder,
        {
            "limit": 1000,
            "offset": 0,
        },
    )

    downloaded = 0

    for item in files:
        name = item.get("name")

        if not name:
            continue

        # Ignore nested folders for now.
        if item.get("id") is None:
            continue

        remote_path = f"{folder}/{name}"
        local_path = local_dir / name

        print(f"Downloading {remote_path}...")

        data = storage.download(remote_path)

        local_path.write_bytes(data)

        downloaded += 1

    return downloaded


def main() -> None:
    client = get_client()

    storage = client.storage.from_(BUCKET)

    # Remove previously downloaded documents.
    # The .gitkeep file is preserved.
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for item in RAW_DIR.iterdir():
        if item.name == ".gitkeep":
            continue

        if item.is_file():
            item.unlink()

        elif item.is_dir():
            shutil.rmtree(item)

    for folder in FOLDERS:
        (RAW_DIR / folder).mkdir(
        parents=True,
        exist_ok=True,
    )

    total = 0

    for folder in FOLDERS:
        count = download_folder(storage, folder)
        print(f"{folder}: {count} files")
        total += count

    if total == 0:
        raise RuntimeError(
            "No documents were downloaded from Supabase."
        )

    print(f"Downloaded {total} documents.")


if __name__ == "__main__":
    main()

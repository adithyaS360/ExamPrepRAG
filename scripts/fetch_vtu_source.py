"""Fetch the publicly available VTU CSE scheme PDF for deployment-time indexing."""

import requests
from pathlib import Path

URL = "https://vtu.ac.in/pdf/2022syll/csesch.pdf"
TARGET = Path("data/raw/vtu_2022_cse_scheme.pdf")


def main() -> None:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    resp = requests.get(URL, headers=headers, timeout=15, verify=True)
    resp.raise_for_status()
    TARGET.write_bytes(resp.content)

    print(f"Downloaded {TARGET} from {URL}")


if __name__ == "__main__":
    main()
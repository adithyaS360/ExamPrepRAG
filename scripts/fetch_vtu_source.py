"""Fetch the public VTU CSE scheme PDF for deployment-time indexing."""
from pathlib import Path
from urllib.request import urlretrieve

URL = "https://vtu.ac.in/pdf/2022syll/csesch.pdf"
TARGET = Path("data/raw/vtu_2022_cse_scheme.pdf")


def main() -> None:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(URL, TARGET)
    print(f"Downloaded {TARGET} from {URL}")


if __name__ == "__main__":
    main()

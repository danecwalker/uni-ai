from pathlib import Path
import shutil
import tempfile
import zipapp


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "dist" / "companion-ai-standalone.pyz"


def main() -> None:
    OUTPUT.parent.mkdir(exist_ok=True)
    if OUTPUT.exists():
        OUTPUT.unlink()

    with tempfile.TemporaryDirectory(prefix="companion-ai-build-") as temp_dir:
        staging = Path(temp_dir) / "app"
        staging.mkdir()
        shutil.copytree(
            ROOT / "companion_ai",
            staging / "companion_ai",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        zipapp.create_archive(
            staging,
            target=OUTPUT,
            interpreter="/usr/bin/env python3",
            main="companion_ai.main:main",
            compressed=True,
        )

    OUTPUT.chmod(0o755)
    print(f"Built {OUTPUT}")


if __name__ == "__main__":
    main()

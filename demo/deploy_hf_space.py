"""Publish the website + live demo as a Hugging Face Docker Space.

    python demo/deploy_hf_space.py --space <user>/<space-name> --token <hf write token>

Stages only what the demo needs (code, scene config, website, the small CPU
detector) with the Space's README header and Dockerfile, then uploads it.
"""
import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["src", "configs", "website", "demo/server.py", "requirements.txt", "requirements-dev.txt",
           "weights/yolo11s.pt"]
SPACE_README = """---
title: Infinity Traffic Events
emoji: 🚦
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

Team Infinity, WIUT Hackathon 2026 (CV track): traffic event detection and accident anticipation.
Code: https://github.com/xurshidbek1812/hackathon
"""


def stage(dst: Path) -> None:
    for rel in INCLUDE:
        src = ROOT / rel
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(src, target)
    (dst / "demo" / "__init__.py").touch()
    shutil.copy2(ROOT / "demo" / "Dockerfile", dst / "Dockerfile")
    (dst / "README.md").write_text(SPACE_README, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True, help="<user>/<space-name>")
    ap.add_argument("--token", required=True)
    args = ap.parse_args()
    api = HfApi(token=args.token)
    api.create_repo(args.space, repo_type="space", space_sdk="docker", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        stage(Path(tmp))
        api.upload_folder(folder_path=tmp, repo_id=args.space, repo_type="space",
                          commit_message="Deploy website and live demo")
    print(f"https://huggingface.co/spaces/{args.space}")
    print(f"https://{args.space.replace('/', '-').replace('_', '-').lower()}.hf.space")


if __name__ == "__main__":
    main()

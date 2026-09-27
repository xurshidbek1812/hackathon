"""Publish the website + live demo as a Hugging Face Docker Space.

    python demo/deploy_hf_space.py --space <user>/<space-name> --token <hf write token> [--static]

Docker Spaces (website + live demo backend) need a paid Hugging Face plan;
--static publishes only the website (free), whose demo section then points
visitors to the local backend (or to `apiBase` in website/config.js).

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


def stage_static(dst: Path) -> None:
    shutil.copytree(ROOT / "website", dst, dirs_exist_ok=True)
    header = SPACE_README.replace("sdk: docker\napp_port: 7860", "sdk: static\napp_file: index.html")
    (dst / "README.md").write_text(header, encoding="utf-8")


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
    ap.add_argument("--static", action="store_true", help="website only (free Space, no demo backend)")
    args = ap.parse_args()
    api = HfApi(token=args.token)
    api.create_repo(args.space, repo_type="space", space_sdk="static" if args.static else "docker", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        (stage_static if args.static else stage)(Path(tmp))
        # switching modes: drop the other layout's files (static puts the site at the root)
        if args.static:
            stale = ["website/*", "src/*", "configs/*", "demo/*", "weights/*", "Dockerfile", "requirements*.txt"]
        else:
            stale = ["index.html", "config.js", "assets/*", "data/*", "tools/*"]
        api.upload_folder(folder_path=tmp, repo_id=args.space, repo_type="space", delete_patterns=stale,
                          commit_message="Deploy website" + ("" if args.static else " and live demo"))
    print(f"https://huggingface.co/spaces/{args.space}")
    host = args.space.replace("/", "-").replace("_", "-").lower()
    print(f"https://{host}.static.hf.space" if args.static else f"https://{host}.hf.space")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Assemble pinned Mia patches and OSSLab ports into a new build directory."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

RECIPE_REF = "c7f2b85100f0a7b099c92a4e21825e30d93c9c51"
ENGINE_REF = "fbbc97132cd004237d9887b8d3185896efaedc5c"
RECIPE_URL = "https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks.git"


def assemble(recipe, custom, output):
    recipe, custom, output = map(Path, (recipe, custom, output))
    ref = subprocess.check_output(["git", "-C", str(recipe), "rev-parse", "HEAD"], text=True).strip()
    if ref != RECIPE_REF:
        raise ValueError("Recipe checkout must match the pinned revision")
    subprocess.run(["git", "-C", str(recipe), "diff", "--exit-code", "HEAD", "--",
                    "zig/patches", "zig/patches-inputs", "scripts/config.sh"], check=True,
                   stdout=subprocess.DEVNULL)
    sources = sorted((recipe / "zig/patches").glob("*.patch")) + sorted(custom.glob("*.patch"))
    names = [p.name for p in sources]
    if len(names) != len(set(names)) or not sources or not list(custom.glob("*.patch")):
        raise ValueError("Missing patches or colliding patch names")
    output.mkdir(parents=True, exist_ok=False)
    engine = output / "patches"
    engine.mkdir()
    for p in sources:
        # Read tracked recipe bytes: an untracked patch in the checkout cannot enter the image.
        if p.is_relative_to(recipe):
            data = subprocess.check_output(["git", "-C", str(recipe), "show", f"HEAD:{p.relative_to(recipe)}"])
        else:
            data = p.read_bytes()
        (engine / p.name).write_bytes(data)
    tracked_inputs = subprocess.check_output(
        ["git", "-C", str(recipe), "ls-files", "-z", "zig/patches-inputs"], text=True).split("\0")
    for name in filter(None, tracked_inputs):
        target = output / Path(name).relative_to("zig")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(subprocess.check_output(["git", "-C", str(recipe), "show", f"HEAD:{name}"]))
    manifest = {"recipe": RECIPE_URL, "recipe_ref": ref, "engine_ref": ENGINE_REF,
                "patches": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(engine.glob("*.patch"))}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recipe", type=Path, required=True)
    p.add_argument("--custom", type=Path, default=Path(__file__).parents[1] / "patches/mia-current")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(assemble(a.recipe, a.custom, a.output), indent=2))


if __name__ == "__main__":
    main()

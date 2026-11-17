import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("assembly", ROOT / "tools/assemble-mia-current.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class AssemblyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.recipe, self.custom = self.root / "recipe", self.root / "custom"
        self.recipe.mkdir()
        self.custom.mkdir()
        for path, data in (("zig/patches/0001-official.patch", "official"),
                           ("zig/patches-inputs/0001-input.patch", "inputs"),
                           ("scripts/config.sh", f'TF_REF="{module.ENGINE_REF}"')):
            p = self.recipe / path
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(data)
        self.git("init", "-q")
        self.git("add", ".")
        self.git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture")
        original = module.RECIPE_REF
        module.RECIPE_REF = self.git("rev-parse", "HEAD").strip()
        self.addCleanup(setattr, module, "RECIPE_REF", original)
        (self.custom / "0100-local.patch").write_text("custom")

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.recipe), *args], text=True)

    def test_assembles_pinned_bytes_and_excludes_untracked_inputs(self):
        (self.recipe / "zig/patches-inputs/untracked.patch").write_text("unexpected")
        out = self.root / "assembled"
        manifest = module.assemble(self.recipe, self.custom, out)
        self.assertEqual((out / "patches/0001-official.patch").read_text(), "official")
        self.assertEqual((out / "patches/0100-local.patch").read_text(), "custom")
        self.assertEqual((out / "patches-inputs/0001-input.patch").read_text(), "inputs")
        self.assertFalse((out / "patches-inputs/untracked.patch").exists())
        self.assertEqual(len(manifest["patches"]), 2)
        self.assertEqual(manifest["engine_ref"], module.ENGINE_REF)
        with self.assertRaises(FileExistsError):
            module.assemble(self.recipe, self.custom, out)

    def test_wrong_pin_dirty_recipe_and_name_collisions_are_refused(self):
        out = self.root / "assembled"
        pin = module.RECIPE_REF
        module.RECIPE_REF = "0" * 40
        with self.assertRaisesRegex(ValueError, "pinned revision"):
            module.assemble(self.recipe, self.custom, out)
        module.RECIPE_REF = pin
        official = self.recipe / "zig/patches/0001-official.patch"
        official.write_text("modified")
        with self.assertRaises(subprocess.CalledProcessError):
            module.assemble(self.recipe, self.custom, out)
        official.write_text("official")
        (self.custom / official.name).write_text("collision")
        with self.assertRaisesRegex(ValueError, "colliding"):
            module.assemble(self.recipe, self.custom, out)
        self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()

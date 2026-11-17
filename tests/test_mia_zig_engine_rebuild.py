import importlib.util
from pathlib import Path
import re
import tempfile
import unittest

ROOT = Path(__file__).parents[1]
module_spec = importlib.util.spec_from_file_location("rebuild", ROOT / "tools/rebuild-mia-zig-engine.py")
module = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(module)


class EngineRebuildTests(unittest.TestCase):
    def test_compile_is_capped_and_low_priority(self):
        argv = module.build_argv("sha256:base", "/p", "/o", "4g", "0-3", "1000:1000")
        self.assertEqual(argv[argv.index("--memory") + 1], "4g")
        self.assertEqual(argv[argv.index("--memory-swap") + 1], "4g")
        self.assertEqual(argv[argv.index("--cpuset-cpus") + 1], "0-3")
        self.assertEqual(argv[argv.index("--cpu-shares") + 1], "2")
        self.assertIn("/o:/opt/tensorfold", argv)
        self.assertIn("/p:/opt/tf-patches:ro", argv)
        self.assertIn(f"TF_REF={module.TF_REF}", argv)
        self.assertEqual(argv[-3:-1], ["sha256:base", "-c"])
        argv_inputs = module.build_argv("sha256:base", "/p", "/o", "3g", "0-3", "1000:1000", inputs_patch_dir="/ip")
        self.assertIn(f"INPUTS_REF={module.INPUTS_REF}", argv_inputs)
        self.assertIn("/ip:/opt/tf-inputs-patches:ro", argv_inputs)

    def test_engine_script_matches_recipe_compile(self):
        script = module.ENGINE_SCRIPT
        self.assertIn('test "$(git rev-parse HEAD)" = "$TF_REF"', script)
        self.assertIn('git apply --check "$p" && git apply "$p"', script)
        self.assertIn("-Doptimize=ReleaseFast", script)
        self.assertIn("--prefix /opt/tensorfold", script)
        self.assertIn("install native", script)

    def test_patches_in_filename_order_and_hash_tracks_content(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ("0007-b.patch", "0001-a.patch"):
                Path(d, name).write_text(name)
            patches = module.patch_files(d)
            self.assertEqual([p.name for p in patches], ["0001-a.patch", "0007-b.patch"])
            first = module.patches_hash(patches)
            Path(d, "0007-b.patch").write_text("changed")
            self.assertNotEqual(first, module.patches_hash(patches))
            self.assertEqual(len(first), 12)

    def test_file_set_allows_only_inputs_left_over(self):
        base = ["native/bin/tensorfold-native", "bin/tf-dsv41-lanes", "bin/tf-dsv41-rank-cache",
                "tools/inputs/make_rope.py", "share/doc/tensorfold/LICENSE-inputs-branch"]
        new = ["native/bin/tensorfold-native", "bin/tf-dsv41-lanes"]
        self.assertEqual(module.check_files(base, new), sorted(new))
        self.assertEqual(module.check_files(base, new + ["bin/tf-dsv41-port"]),
                         sorted(new + ["bin/tf-dsv41-port"]))
        with self.assertRaises(SystemExit):
            module.check_files(base, new + ["bin/extra"])
        with self.assertRaises(SystemExit):
            module.check_files(base, ["native/bin/tensorfold-native"])

    def test_dockerfile_copies_engine_over_base_by_name(self):
        text = module.dockerfile("osslab/mia-zig-ablit:31eea460", {"tf.patches": "abc"})
        self.assertEqual(text.splitlines()[:2], ["FROM osslab/mia-zig-ablit:31eea460", "COPY tensorfold/ /opt/tensorfold/"])
        self.assertIn('LABEL tf.patches="abc"', text)
        with self.assertRaises(ValueError):
            module.dockerfile("sha256:101217f1", {})

    def test_compiled_engine_reused_only_for_same_base_and_patches(self):
        with tempfile.TemporaryDirectory() as d:
            marker, out = Path(d, "engine-built.json"), Path(d, "tensorfold")
            built = {"base_image": "sha256:a", "tf.patches": "abc"}
            self.assertFalse(module.compiled(marker, out, built))
            out.mkdir()
            Path(out, "f").write_text("x")
            marker.write_text(module.json.dumps(built))
            self.assertTrue(module.compiled(marker, out, built))
            self.assertFalse(module.compiled(marker, out, {**built, "tf.patches": "other"}))

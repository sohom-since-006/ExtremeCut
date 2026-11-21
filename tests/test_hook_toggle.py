"""Tests for removing/disabling Big Text (Hook Overlay) completely."""
import unittest
import tempfile
from pathlib import Path
from studio import jobs, render


class TestHookToggle(unittest.TestCase):
    def test_clean_options_show_hook(self):
        opts_off = jobs.clean_options({"show_hook": False})
        self.assertFalse(opts_off["show_hook"])

        opts_on = jobs.clean_options({"show_hook": True})
        self.assertTrue(opts_on["show_hook"])

        # Default options include show_hook=True
        full_opts = {**jobs.DEFAULTS, **jobs.clean_options({})}
        self.assertTrue(full_opts["show_hook"])

    def test_request_render_preserves_show_hook(self):
        with tempfile.TemporaryDirectory() as td:
            job_dir = Path(td)
            (job_dir / "clips").mkdir(parents=True, exist_ok=True)
            clip = {"id": "c1", "start": 0.0, "end": 10.0, "hook": "ORIGINAL HOOK", "show_hook": True}
            job = {"id": "j1", "clips": [clip], "duration": 30.0, "dir": str(job_dir)}
            jobs._jobs["j1"] = job
            try:
                # Re-render with show_hook disabled
                jobs.request_render("j1", "c1", {"show_hook": False})
                self.assertFalse(clip["show_hook"])
                self.assertEqual(clip["hook"], "")

                # Re-render with layout="blur"
                jobs.request_render("j1", "c1", {"layout": "blur"})
                self.assertEqual(clip["layout"], "blur")

                # Re-render with empty hook text
                jobs.request_render("j1", "c1", {"hook": "   ", "show_hook": True})
                self.assertFalse(clip["show_hook"])
                self.assertEqual(clip["hook"], "")

                # Re-render with show_hook enabled and text
                jobs.request_render("j1", "c1", {"show_hook": True, "hook": "NEW HOOK"})
                self.assertTrue(clip["show_hook"])
                self.assertEqual(clip["hook"], "NEW HOOK")
            finally:
                jobs._jobs.pop("j1", None)

    def test_render_build_ass_omits_hook_when_empty(self):
        words = [{"w": "Hello", "s": 0.0, "e": 1.0}]
        ass_with_hook = render.build_ass(words, 0.0, 10.0, "BIG HOOK TITLE", "kinetic", [], 160, 700)
        self.assertIn("BIG HOOK TITLE", ass_with_hook)

        # When show_hook is disabled, hook string is set to empty ""
        ass_no_hook = render.build_ass(words, 0.0, 10.0, "", "kinetic", [], 160, 700)
        self.assertNotIn("BIG HOOK TITLE", ass_no_hook)
        # Ensure no Dialogue lines for Hook style exist
        dialogue_lines = [line for line in ass_no_hook.splitlines() if line.startswith("Dialogue:")]
        hook_lines = [l for l in dialogue_lines if ",Hook," in l]
        self.assertEqual(len(hook_lines), 0)


if __name__ == "__main__":
    unittest.main()

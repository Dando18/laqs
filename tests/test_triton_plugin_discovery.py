from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from relay.triton_frontend import _default_plugin_path


class TritonPluginDiscoveryTests(unittest.TestCase):
    def test_prefers_plugin_built_for_active_triton_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "python" / "triton"
            plugin = package / "plugins" / "libLAQSTritonAccessManifest.so"
            plugin.parent.mkdir(parents=True)
            plugin.touch()
            active = SimpleNamespace(__file__=package / "__init__.py")

            with mock.patch.dict("sys.modules", {"triton": active}):
                self.assertEqual(_default_plugin_path(), plugin.resolve())

    def test_does_not_fall_back_to_another_checkout_abi(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "python" / "triton"
            package.mkdir(parents=True)
            active = SimpleNamespace(__file__=package / "__init__.py")

            with mock.patch.dict("sys.modules", {"triton": active}):
                self.assertIsNone(_default_plugin_path())


if __name__ == "__main__":
    unittest.main()

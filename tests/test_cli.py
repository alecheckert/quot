from click.testing import CliRunner
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from unittest import TestCase
from quot.gui.__main__ import cli as quot_cli
from quot.__main__ import (
    batch_track as quot_track_cli,
    make_naive_config as quot_config_cli,
)


FIXTURES = Path(__file__).absolute().parent / "fixtures"


class TestCli(TestCase):
    def test_quot(self):
        runner = CliRunner()
        result = runner.invoke(quot_cli, ["--help"])
        assert result.exit_code == 0, result.exit_code

    def test_quot_config(self):
        out_dir = TemporaryDirectory()
        out_path = Path(out_dir.name).absolute() / "test_out.yaml"
        assert not out_path.is_file()
        runner = CliRunner()
        result = runner.invoke(quot_config_cli, [str(out_path)])
        assert result.exit_code == 0, result.exit_code
        assert out_path.is_file()

    def test_quot_track(self):
        config_path = FIXTURES / "sample_config.toml"
        assert config_path.is_file(), config_path
        indir = TemporaryDirectory()
        indir_path = Path(indir.name).absolute()
        src_sample_movie = FIXTURES / "sample_movie.tif"
        sample_movie = indir_path / "sample_movie.tif"
        copyfile(str(src_sample_movie), str(sample_movie))
        expected_output = indir_path / "sample_movie_trajs.csv"
        assert not expected_output.is_file()
        runner = CliRunner()
        result = runner.invoke(
            quot_track_cli, [str(indir_path), str(config_path), "--ext", ".tif"]
        )
        assert result.exit_code == 0, result.exit_code
        assert expected_output.is_file()

    def test_quot_config_quot_track(self):
        """Test that config files produced with quot-config
        are viable inputs to quot-track."""
        indir = TemporaryDirectory()
        indir_path = Path(indir.name).absolute()

        dst_config = indir_path / "sample_config.toml"
        src_movie = FIXTURES / "sample_movie.tif"
        dst_movie = indir_path / "sample_movie.tif"

        assert not dst_config.is_file()
        assert src_movie.is_file()
        assert not dst_movie.is_file()

        copyfile(str(src_movie), str(dst_movie))
        runner = CliRunner()
        result = runner.invoke(quot_config_cli, [str(dst_config)])
        assert result.exit_code == 0, result.exit_code
        assert dst_config.is_file()

        result = runner.invoke(
            quot_track_cli, [str(indir_path), str(dst_config), "--ext", ".tif"]
        )
        assert result.exit_code == 0, result.exit_code
        expected = indir_path / "sample_movie_trajs.csv"
        assert expected.is_file()

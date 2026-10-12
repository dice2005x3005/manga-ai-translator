import logging
from pathlib import Path

from manga_ai_translator.logging_setup import LOGGER_NAME, reset_logging, setup_logging


def test_log_file_records_debug_even_when_console_is_info(tmp_path: Path) -> None:
    log_file = setup_logging("INFO", tmp_path / "logs")
    assert log_file is not None
    logging.getLogger(f"{LOGGER_NAME}.test").debug("hidden from console")
    reset_logging()
    assert "hidden from console" in log_file.read_text(encoding="utf-8")


def test_unwritable_logs_dir_falls_back_to_console(tmp_path: Path) -> None:
    not_a_dir = tmp_path / "file.txt"
    not_a_dir.write_text("x", encoding="utf-8")
    assert setup_logging("INFO", not_a_dir) is None


def test_setup_is_idempotent(tmp_path: Path) -> None:
    setup_logging("INFO", tmp_path)
    setup_logging("INFO", tmp_path)
    assert len(logging.getLogger(LOGGER_NAME).handlers) == 2

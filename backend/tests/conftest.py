import os
import shutil
import tempfile
from pathlib import Path

import pytest

TEST_ROOT = Path(tempfile.mkdtemp(prefix="casework-prototype-tests-"))
os.environ["DATABASE_URL"] = f"sqlite:///{(TEST_ROOT / 'test.sqlite3').as_posix()}"
os.environ["STORAGE_DIR"] = str(TEST_ROOT / "private_uploads")
os.environ["APP_API_TOKEN"] = ""


@pytest.fixture(scope="session", autouse=True)
def remove_test_data():
    yield
    shutil.rmtree(TEST_ROOT, ignore_errors=True)


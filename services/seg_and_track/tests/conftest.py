from pathlib import Path
from typing import AsyncGenerator

import pytest
from httpx import AsyncClient

from app import app

BASE_DIR = Path(__file__).resolve().parent.parent

@pytest.fixture(scope="session")
async def ac() -> AsyncGenerator[AsyncClient, None]:
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

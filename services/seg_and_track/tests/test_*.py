import os.path

from httpx import AsyncClient

from conftest import BASE_DIR


async def test_segment(ac: AsyncClient):
    response = await ac.post(
        url="/segment",
        files={"file": open(os.path.join(BASE_DIR, "tests", "data", "images", "1726497193093553580.png"), "rb")},
    )
    assert response.status_code == 200

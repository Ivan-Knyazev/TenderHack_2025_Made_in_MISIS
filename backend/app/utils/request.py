import aiohttp
from typing import Tuple, Dict, Any


async def make_request(url: str, headers: Dict[str, str], data: Dict[str, str]) -> Tuple[int, Dict[str, Any]]:
    # url = "http://localhost:8000/query"
    # headers = {"Content-Type": "application/json"}
    # data = {
    #     "query": "Не могу открыть стартовую страницу, что делать?",
    #     "conversation_id": "2"
    # }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=data) as response:
            status_code = response.status
            response_json = await response.json()

            # print("TEST Status Code:", status_code)
            # print("TEST Response JSON:", response_json)

            return status_code, response_json

import aiohttp


async def make_request(url: str, headers: dict, data: dict):
    # url = "http://localhost:8000/query"
    # headers = {"Content-Type": "application/json"}
    # data = {
    #     "query": "Не могу открыть стартовую страницу, что делать?",
    #     "conversation_id": "2"
    # }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=data) as response:
            # print("Status Code:", response.status)
            response_json = await response.json()
            # print("Response JSON:", response_json)
            return response_json

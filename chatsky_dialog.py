import requests
import uuid
import json
import datetime
import copy
import pathlib


URL = "http://localhost:8099"
# URL = "http://chatsky.ipavlov.mipt.ru:8000"


request = {
    "user_id": str(uuid.uuid4()),
    "payload": {"type": "text_request", "text_request": "Привет, Квант!"},
}
dialog_logs_dir = pathlib.Path("dialog_logs")
dialog_logs_dir.mkdir(parents=True, exist_ok=True)
dialog = []
while True:
    text_request = input("request:")
    if "/save" in text_request:
        dump_file = dialog_logs_dir / f"dialog_{datetime.datetime.now()}.json"
        json.dump(dialog, dump_file.open("w"), ensure_ascii=False, indent=4)
    else:
        request["payload"]["text_request"] = text_request
        try:
            response = requests.post(url=URL, json=request)
            response = response.json()
        except Exception as e:
            response = {"error": str(e)}
        dialog.append(copy.deepcopy({"request": request, "response": response}))
        print(f"{response}")

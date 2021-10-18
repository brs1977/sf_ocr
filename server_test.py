import os
import json
import time
from fastapi.testclient import TestClient
from server import app


client = TestClient(app)


def test_read_main():        
    filename = '/home/ruslan/prj/sf_ocr/input/kTo84xXwdKk0qzkYnx8Ic09qWZELRS.pdf'
    # print(filename)
    with open(filename, mode='rb') as test_file:
        files = {"file": (os.path.basename(filename), test_file, "application/pdf")}
        response = client.post("/ocr", files=files)

        print(response.text)
        data = json.loads(response.text)
        print(data)

    assert response.status_code == 200
    assert response.headers['content-type'] == 'application/json'
    # time.sleep(3)
    response = client.get(f"/progress/{data['id']}")
    print(response.text)
    response = client.get(f"/result/{data['id']}")
    # print(response.text)
    response = client.get(f"/progress/{data['id']}")
    print(response.text)

test_read_main()    
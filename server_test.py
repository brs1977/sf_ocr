import os
import json
import time
from fastapi.testclient import TestClient
from server import app
from concurrent.futures.process import ProcessPoolExecutor

app.state.executor = ProcessPoolExecutor() 
client = TestClient(app)

def read_json(file_name):
    with open(file_name) as f:
        return json.load(f)

def test_read_main():        
    test_data = read_json('test_data/256277.json')
    for i,info in enumerate(test_data):
        del info['files']
        test_data[i] = info


    filename = 'input/256277.pdf'
    # filename = 'input/256611.pdf'
    # filename = 'input/e70e638e-0981-4bda-8515-ee75c25a812e.pdf' # 30 pic 
    # filename = '/home/rttec/projects/sf_ocr/input/6295ad01-652d-4258-b268-170c9aac0503.pdf' #140
    # print(filename)
    with open(filename, mode='rb') as test_file:
        files = {"file": (os.path.basename(filename), test_file, "application/pdf")}
        response = client.post("/ocr", files=files)
        data = json.loads(response.text)

    assert response.status_code == 200
    assert response.headers['content-type'] == 'application/json'
    # time.sleep(3)

    id = data['id']
    response = client.get(f"/progress/{id}")
    data = json.loads(response.text)

    # сравнить с тестом, выкинуть files (случайные названия)
    data = data['results']
    for i,info in enumerate(data):
        del info['files']
        data[i] = info
    assert data == test_data

    response = client.get(f"/result/{id}")
    

    response = client.get(f"/progress/{id}")
    # print(response.text)

test_read_main()    


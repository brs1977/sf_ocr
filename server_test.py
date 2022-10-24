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
    # filename = 'input/d47caae2-f007-4d1c-a452-3b5163ddc25b.pdf' # text
    # print(filename)
    # filename = 'input/0fc5f911-0f28-456e-8419-5a2a8de25804.pdf'
    # filename = 'input/879e9064-600b-4711-8383-006caac9696c.pdf'
    # filename = 'input/61482eb6-cabd-4429-a796-cddd4d1956a3.pdf'
    # filename = 'input/559c6ca1-e0b3-4a11-91b2-7084fd14a14c.pdf'
    # filename = 'input/7d1a7165-5e6c-4d6e-b87a-d0cc306ff4b4.pdf'
    
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


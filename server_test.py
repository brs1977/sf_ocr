import os
import json
# from fastapi.testclient import TestClient
from server import app
# from concurrent.futures.process import ProcessPoolExecutor
from loguru import logger
from pdf_splitter import PDFSplitter



def read_json(file_name):
    with open(file_name) as f:
        return json.load(f)

def test_www():        
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

def test_split():
    try:
        results = []
        # pdf_file_name = './input/ттттттт.pdf'
        # zip_file_name = './output/ттттттт.zip'

        # pdf_file_name = './input/с-ф 1.pdf'
        # zip_file_name = './output/с-ф 1.zip'

        pdf_file_name = './input/ЦР-933.pdf' # не правильно определяет 2ю страницу инверсия
        zip_file_name = './output/ЦР-933.zip'

        pdf_file_name = './input/873b6d73-8705-4f8f-8ae8-8bc4dd08d43e.pdf' # не правильно определяет 2ю страницу 
        zip_file_name = './output/873b6d73-8705-4f8f-8ae8-8bc4dd08d43e.zip'

        splitter = PDFSplitter(zip_file_name, pdf_file_name,
                               app.orient_clf, app.type_clf, app.extractor)

        for page, pages, info in splitter.process():            
            results.append(info)
            state = {'page': page, 'pages': pages}

        state['results'] = results
        state['url'] = f'result/{id}'

    except Exception as e:
        logger.exception(e)
        state = {'detail': str(e)}


if __name__ == '__main__':
    # app.state.executor = ProcessPoolExecutor() 
    # client = TestClient(app)

    # test_www()    
    test_split()
# [{"sf_no": "265", "sf_date": "28.01.2625", "buyer_inn": null, "buyer_kpp": null, "seller_inn": null, "seller_kpp": null, "files": ["d56owumq-1.pdf"]}, 
#  {"sf_no": "5776", "sf_date": "27.12.2024", "buyer_inn": "7714077682", "buyer_kpp": "771401001", "seller_inn": "7611023357", "seller_kpp": "761101001", "files": ["w_kh8tk2-1.pdf", "w_kh8tk2-2.pdf"]}, 
#  {"sf_no": "83", "sf_date": "16.01.2025", "buyer_inn": "7730239877", "buyer_kpp": "330543001", "seller_inn": "7611023357", "seller_kpp": "761101001", "files": ["smavv7lx-1.pdf", "smavv7lx-2.pdf", "smavv7lx-3.pdf", "smavv7lx-4.pdf"]}]
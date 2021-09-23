import uuid
from zipfile import ZipFile
from io import BytesIO
import os
import sys

if sys.version_info[:2] >= (3, 7):
    from asyncio import get_running_loop
else:
    from asyncio import _get_running_loop as get_running_loop
from fastapi.responses import FileResponse
from fastapi import Request, FastAPI, File, UploadFile, HTTPException  
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import aiofiles
import pickle
import json
from config import Config
from extractor import SfInfoExtractor
from hog_classifier import load_model
from pdf_utils import split_pdf_gen
import re

INDEX_PATTERN = re.compile('_(\d*)')

def del_state(id):
    os.remove(os.path.join(app.file_path,f'{id}.pkl'))

def set_state(id, page, pages, url = None):
    state = {'page':page, 'pages': pages, 'url': url}
    with open(os.path.join(app.file_path,f'{id}.pkl'), 'wb') as f:   
        pickle.dump(state, f)

def get_state(id):
    try:
        with open(os.path.join(app.file_path,f'{id}.pkl'), 'rb') as f:   
            state = pickle.load(f)
    except:
        raise HTTPException(status_code=404)
    return state

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

app.file_path = 'output'
config = Config('models/config.yaml')
app.extractor = SfInfoExtractor(config)
app.orient_clf = load_model('models/orient.pkl')
app.type_clf = load_model('models/type.pkl')    



def get_file_name(fn, files):
    
    if not fn:
        fn = 'undefined'
    else:
        fn = fn.replace('/','_')
    file_name = fn

    while fn in files.keys():
        idx = INDEX_PATTERN.findall(fn)
        if idx:
            idx = idx[-1]
        if not idx:
            idx = 0
        idx = int(idx) + 1
        fn = file_name + '_' + str(idx)
        
    return fn


async def do_work(id):    
    files = {}
    pdf_file_name = os.path.join(app.file_path,f'{id}.pdf')
    zip_file_name = os.path.join(app.file_path,f'{id}.zip')
    with BytesIO() as archive:
        with ZipFile(archive, 'w') as zip_archive:

            for page,pages,info,pdf in split_pdf_gen(pdf_file_name,app.orient_clf,app.type_clf,app.extractor):
                file_name = get_file_name(info['sf_no'], files)
                files[file_name] = file_name

                with zip_archive.open(file_name+'.pdf', 'w') as pdf_file:                    
                    pdf_file.write(pdf.tobytes(garbage=4, deflate=True))

                
                with zip_archive.open(file_name+'.json', 'w') as json_file:
                    json_file.write(bytes(json.dumps(info),'utf-8'))
                    # json.dump(info, json_file)

                set_state(id,page,pages)

        with open(zip_file_name, 'wb') as f:
            f.write(archive.getbuffer())

        set_state(id,page,pages,f'result/{id}.zip')

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):    
    id = str(uuid.uuid4())
    set_state(id=id,page=0,pages=0)

    out_file_name = f"output/{id}.pdf"
    async with aiofiles.open(out_file_name, 'wb') as out_file:
        content = await file.read()  # async read
        await out_file.write(content)  # async write
    
    asyncio.run_coroutine_threadsafe(do_work(id), loop=get_running_loop())

    return JSONResponse({"id": id})


@app.get('/progress/{id}')
async def status(id):
    state = get_state(id)
    return JSONResponse(state)

@app.get('/result/{id}')
async def status(id):
    state = get_state(id)
    if state['page']!=0 and state['page']!=state['pages']:
        raise HTTPException(status_code=404)
    
    del_state(id)
    os.remove(os.path.join(app.file_path,f'{id}.pdf'))

    return FileResponse(path=os.path.join(app.file_path, f'{id}.zip'), media_type='application/zip')
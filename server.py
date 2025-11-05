import os
import sys
import fcntl
from pathlib import Path
import threading
from concurrent.futures.process import ProcessPoolExecutor
from http import HTTPStatus
from io import BytesIO
from threading import Lock
from uuid import UUID, uuid4
from zipfile import ZipFile

from loguru import logger

from pdf_splitter import PDFSplitter

if sys.version_info[:2] >= (3, 7):
    from asyncio import get_running_loop
else:
    from asyncio import _get_running_loop as get_running_loop

import asyncio
import json
import pickle
import re


import aiofiles
from fastapi import (BackgroundTasks, FastAPI, File, HTTPException, Request,
                     UploadFile)
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from config import Config
from extractor import SfInfoExtractor
from hog_classifier import load_model
from pdf_splitter import PDFSplitter

INDEX_PATTERN = re.compile('_(\d*)')


app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=[
                   "*"], allow_methods=["*"], allow_headers=["*"])

app.file_path = 'output'
FILE_PATH = Path("output").resolve() 
config = Config('models/config.yaml')
app.extractor = SfInfoExtractor(config)
app.orient_clf = load_model('models/orient.pkl')
app.type_clf = load_model('models/type.pkl')


# app.lock = Lock()
# app.lock = threading.Lock()

def del_state(id):
    state_file = os.path.join(FILE_PATH, f'{id}.pkl')
    if not os.path.exists(state_file):
        raise FileNotFoundError(f"Файл состояния не найден: {state_file}")
    
    with open(state_file, "rb") as f:        
        os.remove(state_file)


# def set_state(id, state):
#     with app.lock:
#         with open(os.path.join(app.file_path, f'{id}.pkl'), 'wb') as f:
#             pickle.dump(state, f)

# def set_state(id, new_state):
#     state_file = os.path.join(FILE_PATH, f'{id}.pkl')
#     if not state_file.exists():
#         raise FileNotFoundError(f"Файл состояния не найден: {state_file}")

#     # Считаем текущее состояние из файла, если есть
#     try:
#         with open(state_file, 'rb') as f:
#             current_state = pickle.load(f)
#     except (FileNotFoundError, EOFError, pickle.UnpicklingError):
#         current_state = {'page': 0, 'pages': 0}

#     # Обновляем состояние только если новое впереди текущего
#     if (new_state.get('page', 0) > current_state.get('page', 0) or
#         new_state.get('pages', 0) > current_state.get('pages', 0)):
#         with open(state_file, 'wb') as f:
#             pickle.dump(new_state, f)

def set_state(id: str, new_state: dict):
    state_file = os.path.join(FILE_PATH, f'{id}.pkl')
    lock_file = os.path.join(FILE_PATH, f'{id}.lock')
    logger.debug([id, new_state])

    with open(lock_file, "w") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)  # эксклюзивная блокировка
        try:
            # Загружаем текущее состояние (если файл существует)
            if os.path.exists(state_file):
                try:
                    with open(state_file, "rb") as f:
                        current_state = pickle.load(f)
                except (EOFError, pickle.UnpicklingError):
                    current_state = {"page": 0, "pages": 0}
            else:
                current_state = {"page": 0, "pages": 0}

            # Обновляем только если новое состояние "вперёд"
            if (not os.path.exists(state_file) or 
                new_state.get("page", 0) > current_state.get("page", 0) or
                new_state.get("pages", 0) > current_state.get("pages", 0)):
                with open(state_file, "wb") as f:
                    pickle.dump(new_state, f)
        finally:
            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)  # снимаем блокировку

def get_state(id):
    state_file = os.path.join(FILE_PATH, f'{id}.pkl')
    if not os.path.exists(state_file):
        raise FileNotFoundError(f"Файл состояния не найден: {state_file}")
    with open(state_file, "rb") as f:
        return pickle.load(f)

def get_file_name(fn, files):

    if not fn:
        fn = 'undefined'
    else:
        fn = fn.replace('/', '_')
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

 
def do_work(id):
    try:
        results = []
        pdf_file_name = os.path.join(app.file_path, f'{id}.pdf')
        zip_file_name = os.path.join(app.file_path, f'{id}')

        splitter = PDFSplitter(zip_file_name, pdf_file_name,
                               app.orient_clf, app.type_clf, app.extractor)

        state = get_state(id)
        for page, pages, info in splitter.process():            
            results.append(info)
            state = {'page': page, 'pages': pages}
            logger.debug(state)
            set_state(id, state)

        state['results'] = results
        state['url'] = f'result/{id}'

    except Exception as e:
        logger.exception(e)
        state = {'detail': str(e)}
    return state


# def do_work(id):
#     try:
#         files = {}
#         results = []
#         pdf_file_name = os.path.join(app.file_path,f'{id}.pdf')
#         zip_file_name = os.path.join(app.file_path,f'{id}.zip')
#         with BytesIO() as archive:
#             with ZipFile(archive, 'w') as zip_archive:

#                 for page,pages,info,pdf in split_pdf_gen(pdf_file_name,app.orient_clf,app.type_clf,app.extractor):
#                     file_name = get_file_name(info['sf_no'], files)
#                     files[file_name] = file_name

#                     with zip_archive.open(file_name+'.pdf', 'w') as pdf_file:
#                         pdf_file.write(pdf.tobytes(garbage=4, deflate=True))


#                     with zip_archive.open(file_name+'.json', 'w') as json_file:
#                         json_file.write(bytes(json.dumps(info),'utf-8'))
#                         # json.dump(info, json_file)
#                     res = {'json':file_name+'.json', 'file': file_name+'.pdf', 'data': info, 'raw_data': info}
#                     results.append(res)

#                     state = {'page':page,'pages':pages}
#                     set_state(id,state)

#             with open(zip_file_name, 'wb') as f:
#                 f.write(archive.getbuffer())
#         state['results'] = results
#         state['url'] = f'result/{id}.zip'

#     except Exception as e:
#         logger.exception(e)
#         state = {'detail':str(e)}
#     return state


async def run_in_process(fn, *args):
    loop = asyncio.get_event_loop()
    state = await loop.run_in_executor(app.state.executor, fn, *args)
    logger.debug([run_in_process, state])
    set_state(id, state)


async def cpu_bound_task(id: str) -> None:
    await run_in_process(do_work, id)
    


@app.post("/ocr", status_code=HTTPStatus.ACCEPTED)
async def ocr(background_tasks: BackgroundTasks, file: UploadFile = File(...)):

    id = str(uuid4())

    set_state(id, {'page': 0, 'pages': 0})

    # save input pdf file
    out_file_name = f"output/{id}.pdf"
    async with aiofiles.open(out_file_name, 'wb') as out_file:
        content = await file.read()  # async read
        await out_file.write(content)  # async write

    background_tasks.add_task(cpu_bound_task, id)
    return JSONResponse({"id": id})


@app.get('/progress/{id}')
def status(id):
    try:
        state = get_state(id)
        return state
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/result/{id}')
def result(id):
    try:
        #TODO state c ошибкой в нем нет page
        state = get_state(id)
        if (state['page'] == 0 or state['page'] != state['pages']):
            raise HTTPException(status_code=404)

        # del_state(id)
        # os.remove(os.path.join(app.file_path, f'{id}.pdf'))

        return FileResponse(path=os.path.join(app.file_path, f'{id}.zip'), media_type='application/zip')
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("startup")
async def startup_event():
    app.state.executor = ProcessPoolExecutor(max_workers=1)


@app.on_event("shutdown")
async def on_shutdown():
    app.state.executor.shutdown()

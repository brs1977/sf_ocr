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

def del_state(_id):
    state_file = os.path.join(FILE_PATH, f'{_id}.pkl')
    if not os.path.exists(state_file):
        raise FileNotFoundError(f"Файл состояния не найден: {state_file}")
    
    os.remove(state_file)

def set_state(_id: str, new_state: dict):
    state_file = os.path.join(FILE_PATH, f'{_id}.pkl')
    lock_file = os.path.join(FILE_PATH, f'{_id}.lock')
    logger.debug(["set_state", _id, new_state])

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

def get_state(_id):
    state_file = os.path.join(FILE_PATH, f'{_id}.pkl')
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

 
def do_work(_id):
    try:
        results = []
        pdf_file_name = os.path.join(app.file_path, f'{_id}.pdf')
        zip_file_name = os.path.join(app.file_path, f'{_id}')

        splitter = PDFSplitter(zip_file_name, pdf_file_name,
                               app.orient_clf, app.type_clf, app.extractor)

        state = get_state(_id)
        for page, pages, info in splitter.process():            
            results.append(info)
            state = {'page': page, 'pages': pages}
            logger.debug(state)
            set_state(_id, state)

        state['results'] = results
        state['url'] = f'result/{_id}'

    except Exception as e:
        logger.exception(e)
        state = {'detail': str(e)}
    return state

async def run_in_process(fn, *args):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(app.state.executor, fn, *args)
    


async def cpu_bound_task(_id: str) -> None:
    state = await run_in_process(do_work, _id)
    set_state(_id, state)


@app.post("/ocr", status_code=HTTPStatus.ACCEPTED)
async def ocr(background_tasks: BackgroundTasks, file: UploadFile = File(...)):

    _id = str(uuid4())

    set_state(_id, {'page': 0, 'pages': 0})

    # save input pdf file
    out_file_name = f"output/{_id}.pdf"
    async with aiofiles.open(out_file_name, 'wb') as out_file:
        content = await file.read()  # async read
        await out_file.write(content)  # async write

    background_tasks.add_task(cpu_bound_task, _id)
    return JSONResponse({"id": _id})


@app.get('/progress/{_id}')
def status(_id):
    try:
        state = get_state(_id)
        return state
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/result/{_id}')
def result(_id):
    try:
        #TODO state c ошибкой в нем нет page
        state = get_state(_id)
        if (state['page'] == 0 or state['page'] != state['pages']):
            raise HTTPException(status_code=404)

        return FileResponse(path=os.path.join(app.file_path, f'{_id}.zip'), media_type='application/zip')
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("startup")
async def startup_event():
    app.state.executor = ProcessPoolExecutor(max_workers=1)


@app.on_event("shutdown")
async def on_shutdown():
    app.state.executor.shutdown()

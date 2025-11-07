import os
import sys

from lock import lock_file, unlock_file

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

from pathlib import Path
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

config = Config('models/config.yaml')
OUTPUT_DIR = Path("output").resolve()
OUTPUT_DIR.mkdir(exist_ok=True)
ORIENT_CLF = load_model('models/orient.pkl')
TYPE_CLF = load_model('models/type.pkl')
EXTRACTOR = SfInfoExtractor(config)


# app.file_path = 'output'
# app.extractor = SfInfoExtractor(config)
# app.orient_clf = load_model('models/orient.pkl')
# app.type_clf = load_model('models/type.pkl')


# app.lock = Lock()
# app.lock = threading.Lock()

# def del_state(_id):
#     with app.lock:
#         os.remove(os.path.join(app.file_path, f'{_id}.pkl'))

# def set_state(id, state):
#     with app.lock:
#         with open(os.path.join(app.file_path, f'{id}.pkl'), 'wb') as f:
#             pickle.dump(state, f)

# def set_state(_id, new_state):
#     with app.lock:
#         file_path = os.path.join(app.file_path, f'{_id}.pkl')
#         # Считаем текущее состояние из файла, если есть
#         try:
#             with open(file_path, 'rb') as f:
#                 current_state = pickle.load(f)
#         except (FileNotFoundError, EOFError, pickle.UnpicklingError):
#             current_state = {'page': 0, 'pages': 0}

#         # Обновляем состояние только если новое впереди текущего
#         if (new_state.get('page', 0) > current_state.get('page', 0) or
#             new_state.get('pages', 0) > current_state.get('pages', 0)):
#             with open(file_path, 'wb') as f:
#                 pickle.dump(new_state, f)

# def get_state(_id):
#     with app.lock:
#         with open(os.path.join(app.file_path, f'{_id}.pkl'), 'rb') as f:
#             state = pickle.load(f)
#     return state




def set_state(_id, new_state):
    file_path = OUTPUT_DIR / f"{_id}.pkl"
    lock_path = OUTPUT_DIR / f"{_id}.lock"

    with open(lock_path, "w") as locking_file:
        lock_file(locking_file)
        try:
            # Загружаем текущее состояние
            if file_path.exists():
                try:
                    with open(file_path, "rb") as f:
                        current_state = pickle.load(f)
                except (EOFError, pickle.UnpicklingError):
                    current_state = {"page": 0, "pages": 0}
            else:
                current_state = {"page": 0, "pages": 0}

            # для финального состояния (с results) — всегда сохраняем!
            if 'results' in new_state or 'detail' in new_state:
                with open(file_path, "wb") as f:
                    pickle.dump(new_state, f)
            else:
                # Промежуточное — только если вперёд
                if (not file_path.exists() or
                    new_state.get('page', 0) > current_state.get('page', 0) or
                    new_state.get('pages', 0) > current_state.get('pages', 0)):
                    with open(file_path, "wb") as f:
                        pickle.dump(new_state, f)
        finally:
            unlock_file(locking_file)

def get_state(_id):
    file_path = OUTPUT_DIR / f"{_id}.pkl"
    if not file_path.exists():
        return {"page": 0, "pages": 0}
    with open(file_path, "rb") as f:
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
        pdf_file = OUTPUT_DIR / f"{_id}.pdf"
        zip_dir = OUTPUT_DIR / f"{_id}"

        splitter = PDFSplitter(
            str(zip_dir),
            str(pdf_file),
            ORIENT_CLF,
            TYPE_CLF,
            EXTRACTOR
        )
        
        page = pages = 0
        for page, pages, info in splitter.process():
            results.append(info)
            set_state(_id, {"page": page, "pages": pages})

        final_state = {
            "page": pages,
            "pages": pages,
            "results": results,
            "url": f"result/{_id}"
        }
        set_state(_id, final_state)
        return final_state

    except Exception as e:
        logger.exception(e)
        return {"detail": str(e)}

# def do_work(_id):
#     try:
#         results = []
#         pdf_file_name = os.path.join(app.file_path, f'{_id}.pdf')
#         zip_file_name = os.path.join(app.file_path, f'{_id}')

#         splitter = PDFSplitter(zip_file_name, pdf_file_name,
#                                app.orient_clf, app.type_clf, app.extractor)

#         state = get_state(_id)
#         for page, pages, info in splitter.process():            
#             results.append(info)
#             state = {'page': page, 'pages': pages}
#             set_state(_id, state)

#         state['results'] = results
#         state['url'] = f'result/{_id}'

#     except Exception as e:
#         logger.exception(e)
#         state = {'detail': str(e)}
#     return state


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
    return await loop.run_in_executor(app.state.executor, fn, *args)


async def cpu_bound_task(id: str) -> None:
    state = await run_in_process(do_work, id)
    set_state(id, state)


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

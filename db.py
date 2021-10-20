import sqlite3
import json
from fastapi import HTTPException

db_uri = 'file:cache_database?mode=memory&cache=shared'
timeout = 3

def create_db():
    conn = connect()
    conn.execute('CREATE TABLE IF NOT EXISTS cache (sid text,state text);')
    conn.commit()
    return conn

def connect():
    # return sqlite3.connect(db_uri, timeout=timeout, uri=True, check_same_thread=False)    
    return sqlite3.connect('file::memory:?cache=shared', uri=True, timeout=timeout, isolation_level=None, check_same_thread=False)

def get_state(id):
    conn = connect()
    cursor = conn.cursor()
    cursor.execute("select state from cache where sid = ?",(id,))
    if cursor.rowcount < 0:
        raise HTTPException(status_code=404)
    state = cursor.fetchone()[0]
    state = json.loads(state)
    cursor.close()
    return state

def set_state(id,state):
    state = json.dumps(state)
    conn = connect()
    conn.execute("insert into cache (sid,state) values(?,?)",(id,state))
    conn.commit()

def del_state(id):    
    conn = connect()
    conn.execute("delete from cache where sid = ?",(id,))
    conn.commit()


def test():
    create_db()
    id = 'sdofh'
    state = {'page':1,'pages':10}
    set_state(id,state)
    test_state = get_state(id)
    assert test_state == state
    del_state(id)
    test_state = get_state(id)


if __name__ == '__main__':
    test()
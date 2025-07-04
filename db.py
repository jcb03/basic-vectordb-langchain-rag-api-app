import sqlite3

def init_db():
    conn = sqlite3.connect('metadata.db')
    c=conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS docs
                 (id INTEGER PRIMARY KEY, title TEXT, chunk TEXT)''')
    conn.commit()
    return conn

def insert_chunk(conn, title, chunk):
    c=conn.cursor()
    c.execute('INSERT INTO docs (title, chunk) VALUES (?, ?)', (title, chunk))
    conn.commit()
    return c.lastrowid

def get_chunks_by_ids(conn, ids):
    c=conn.cursor()
    q="SELECT id, title, chunk FROM docs WHERE id IN ({})".format(','.join(['?']*len(ids)))
    c.execute(q, ids)
    return c.fetchall()

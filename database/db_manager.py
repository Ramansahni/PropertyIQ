import sqlite3
import bcrypt
import os
from config.constants import DB_PATH

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    
    # Try adding is_admin column if it doesn't exist
    try:
        c.execute('ALTER TABLE users ADD COLUMN is_admin INTEGER DEFAULT 0')
    except sqlite3.OperationalError:
        pass # Column already exists
        
    # Check if default admin exists
    c.execute('SELECT * FROM users WHERE email = ?', ('admin@propertyiq.com',))
    if not c.fetchone():
        c.execute('INSERT INTO users (name, email, password, is_admin) VALUES (?, ?, ?, ?)',
                  ('Admin', 'admin@propertyiq.com', hash_password('admin123'), 1))
        
    conn.commit()
    conn.close()

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def create_user(name, email, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute('INSERT INTO users (name, email, password, is_admin) VALUES (?, ?, ?, 0)',
                  (name, email, hash_password(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def verify_user(email, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Ensure backwards compatibility if old DB didn't get migrated for some reason
    try:
        c.execute('SELECT id, name, password, is_admin FROM users WHERE email = ?', (email,))
        user = c.fetchone()
    except sqlite3.OperationalError:
        # Fallback if is_admin is missing
        c.execute('SELECT id, name, password FROM users WHERE email = ?', (email,))
        row = c.fetchone()
        user = (row[0], row[1], row[2], 0) if row else None
    finally:
        conn.close()
    
    if user and check_password(password, user[2]):
        return {'id': user[0], 'name': user[1], 'email': email, 'is_admin': user[3]}
    return None

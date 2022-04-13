'''
Created on 04.03.2022

@author: larsw
'''
import mysql.connector
import time
from threading import RLock

class MySQLWrapper():
    def __init__(self, host=None, user=None, password=None, database=None,
                 attempts=10, attempt_wait=1, commit=True):
        self._host = host
        self._user = user
        self._password = password
        self._database = database
        
        self._attempts = attempts
        self._attempt_wait = attempt_wait
        
        self._commit = commit
        
        print("Commit: "+str(commit))
        
        self._lock = RLock()
        self._con = None
        
    def __call__ (self, host=None, user=None, password=None, database=None,
                 attempts=None, attempt_wait=None, commit=None):
        host = self._host if host is None else host
        user = self._user if user is None else user
        password = self._password if password is None else password
        database = self._database if database is None else database
        
        attempts = self._attempts if attempts is None else attempts
        attempt_wait = self._attempt_wait if attempt_wait is None else attempt_wait
        
        commit = self._commit if commit is None else commit
        
        wrapper = MySQLWrapper(host=host, user=user, password=password, database=database,
                               attempts=attempts, attempt_wait=attempt_wait, commit=commit)
        return wrapper
        
    def __enter__ (self):
        self._lock.acquire()
        last_attempt = self._attempts - 1
        
        for attempt in range(self._attempts):
            try:
                self._con = mysql.connector.connect(
                        host=self._host,
                        user=self._user,
                        password=self._password,
                        database=self._database
                    )
                return self._con.cursor()
            except Exception as e:
                if attempt != last_attempt:                        
                    time.sleep(self._attempt_wait)
                else:
                    raise e
                
    def __exit__ (self, exc_type, exc_val, exc_tb):
        if self._commit:
            self._con.commit()
            
        self._con.close()
        self._lock.release()
        
        
        
        
        
        
        
if __name__ == "__main__":
    wrapper = MySQLWrapper(host="localhost", user="root", password="p0k3m0nBlttgrn", database="idealo_data")
    
    query = "SELECT * FROM category;"
    
    with wrapper(commit=False) as cur:
        cur.execute(query)
        rows = cur.fetchall()
        
    print(rows)
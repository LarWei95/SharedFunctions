'''
Created on 17.06.2021

@author: larsw
'''
import os
import os.path as osp
import datetime as dt
import pyautogui
from threading import RLock

DEFAULT_DATETIME_FORMAT = "%Y-%m-%d---%H-%M-%S-%f"

def save_screencap (path, screencap):
    screencap.save(path)

class Screencapper(object):
    '''
    classdocs
    '''
    def __init__(self, save_directory, pool):
        self._pool = pool
        
        if save_directory is not None:
            save_directory = osp.abspath(save_directory)
            os.makedirs(save_directory, exist_ok=True)
            
            self.has_directory = True
        else:
            self.has_directory = False
            
        self._save_directory = save_directory
                
    def screencap_pyautogui (self, save_over_pool=True):
        try:
            screencap = pyautogui.screenshot()
            
            t = dt.datetime.now()
            
            if self.has_directory:
                filename = t.strftime(DEFAULT_DATETIME_FORMAT)+".png"
                path = osp.join(self._save_directory, filename)
                
                if save_over_pool:
                    self._pool.apply_async(save_screencap, (path, screencap))
                else:
                    save_screencap(path, screencap)
                
            return screencap, t
        except:
            print("Fuck")
            return None, None
        
    def start_recording (self):
        pyautogui.keyDown("alt")
        pyautogui.press("f9")
        pyautogui.keyUp("alt")
        
    def stop_recording (self):
        pyautogui.keyDown("alt")
        pyautogui.press("f9")
        pyautogui.keyUp("alt")
        
    def get_available_by_datetimes (self):
        files = [
                x
                for x in os.listdir(self._save_directory)
                if ".png" in x
            ]
        
        availables = []
        
        for file in files:
            file = file.replace(".png", "")
            
            try:
                file = dt.datetime.strptime(file, DEFAULT_DATETIME_FORMAT)
                availables.append(file)
            except Exception:
                continue
            
        return availables
        
class Keylogger ():
    def __init__ (self, save_path):
        self._log_lock = RLock()
        
        save_path = osp.abspath(save_path)
        
        self._log = open(save_path, "w")
        
    def _save_key (self, key, timestamp, actiontype):
        line = "{:s}|{:s}|".format(
                timestamp.strftime(DEFAULT_DATETIME_FORMAT),
                actiontype
            )
        line = line+str(key)+"\n"
        
        self._log.write(line)
        self._log.flush()
        
        
    def add_press (self, key):        
        timestamp = dt.datetime.now()
        
        self._log_lock.acquire(blocking=True)
        
        self._save_key(key, timestamp, "P")     
        
        self._log_lock.release()
            
    def add_release (self, key):
        timestamp = dt.datetime.now()
        
        self._log_lock.acquire(blocking=True)
        
        self._save_key(key, timestamp, "R")     
        
        self._log_lock.release()
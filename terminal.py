'''
Created on 02.06.2021

@author: larsw
'''
import subprocess

class Terminal(object):
    '''
    classdocs
    '''

    def __init__(self):
        self.__shell = subprocess.Popen("C:\\WINDOWS\\system32\\cmd.exe", 
                                        stdin=subprocess.PIPE, 
                                        stdout=subprocess.PIPE, 
                                        stderr=subprocess.PIPE)
        
    def reinitialize_terminal (self):
        if self.__shell is not None:
            self.__shell.close()
            
    
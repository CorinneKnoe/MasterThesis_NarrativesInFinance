# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import multiprocessing

class Plsa(object):
    def __init__(self, repetitions):
        self.no = repetitions
        
    def worker(self, procnum, return_dict):
        '''worker function'''
        return_dict[procnum] = procnum * self.no






if __name__ == "__main__":
    
    plsa = Plsa(2)
    
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    for i in ['anna', 'beta', 'theta', 'delta']:
        p = multiprocessing.Process(target=plsa.worker, args=(i, return_dict))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()
    print(return_dict)
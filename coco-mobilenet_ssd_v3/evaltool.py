import os
import psutil
import threading
import time
import io

class FrameStat:
    startTime = time.time()
    endTime = 0
    fps = 0

    def calTime(self):
        self.endTime = time.time()
        self.fps = round(1/(self.endTime - self.startTime), 1)
        self.startTime = time.time()

    def printFPS(self):
        print(self.fps)
    
    @property
    def FPS(self):
        return self.fps

class CpuStat:
    pid = os.getpid()
    py = psutil.Process(pid)

    @property
    def CpuStat(self):
        return self.py.cpu_percent() / psutil.cpu_count()

    def printCPU(self):
        print(self.py.cpu_percent())

    def printAvgCPU(self):
        print(self.py.cpu_percent() / psutil.cpu_count())    

    def printAvgCPUSec(self, sec):
        self.printAvgCPU()
        threading.Timer(sec, self.printAvgCPUSec, [sec]).start()

class FileStat(CpuStat, FrameStat):
    def __init__(self, fileName):
        self.fileName = fileName
        self.f = open(self.fileName, 'a')
        self.f.write(time.strftime('%c', time.localtime(time.time())) + "\n")
        self.keyDict = None

    def fileOpen(self):
        self.f = open(self.fileName, 'a')

    def fileClose(self):
        if isinstance(self.f, io.TextIOWrapper):
            self.f.close()

    def fstatWrite(self, sec = 1, **kwargs):
        self.fileOpen()
        if isinstance(self.f, io.TextIOWrapper):
            self.f.write(f'PID: {self.pid}  FPS: {self.FPS / sec}  CPU: {self.CpuStat}  ')
            if kwargs:
                for dictKey, dictItem in kwargs.items():
                    self.f.write(f'{dictKey}: {dictItem}  ')
            self.f.write('\n')
        self.fileClose()

    def fstatWriteSec(self, sec):
        if isinstance(self.f, io.TextIOWrapper):
            self.fstatWrite()
            threading.Timer(sec, self.fstatWriteSec, [sec]).start()

    def setKwargs(self, **kwargs):
        self.keyDict = kwargs

    def getKwargs(self):
        return self.keyDict

    def fstatWriteSecArgs(self, sec):
        kwarg = self.keyDict
        if isinstance(self.f, io.TextIOWrapper):
            if kwarg != None:
                self.fstatWrite(**kwarg)
            else:
                self.fstatWrite()
            threading.Timer(sec, self.fstatWriteSecArgs, [sec]).start()

    

if __name__ == "__main__":
    # a = CpuStat()
    # a.printAvgCPUSec(3)

    # b = FrameStat()
    # for i in range(5000000):
    #     i = i
    # b.printFPS()

    c = FileStat("aaaaaaaaaaa.txt")
    for i in range(5000000):
        i = i
    c.calTime()
    c.setKwargs(test='tset')
    c.fstatWriteSecArgs(2)
    i = 0
    while i < 15:
        c.setKwargs(test='a'*i)
        i = i + 1
    c.fileClose()

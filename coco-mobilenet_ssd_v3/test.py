import numpy as np
import threading

class Test:
    print("TEST")

    def __init__(self):
        self.abc = "123"
        self.a = 1
        self.b = 2
        self.c = 3

class Testt:
    print("TEST")

    def __init__(self):
        self.cba = "cba"
        self.q = 1
        self.w = 2
        self.e = 3


class TestSang(Test, Testt):

    def __init__(self):
        Test.__init__(self)
        Testt.__init__(self)
        print(self.abc)
        self.a = 2
        print(self.a)
        print(self.w)

a = np.array([[0, 1], [3, 4], [5, 6]])
b = np.array([0,1,2,3,4])
print(a)
print(a[0])
print(list(np.array(a).reshape(1,-1)[0]))
print(list(np.array(b).reshape(1,-1)[0]))

data = f'{a}'
print(data)

f = open("TESTOBJECT.txt", 'r')
print(type(f))

print(type(1))

a = TestSang()

b = {'aa' : '11', 'bb' : '22'}
print(type(b.items()))


def tesit(**di):
    print('tesit start')
    print(di)
    print('threading?')
    threading.Timer(1, tesit, kwargs=di).start()


tesit(**b)
for i in range(5):
    b['bb'] = i
    print(b)
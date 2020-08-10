from multiprocessing import Pool, Process
from threading import Thread
import time

number = 0

def square(x):
    global number
    number += x
    return x**2

def numbers():
    for i in range(1,10):
        yield i

def print_result():
    with Pool(4) as p:   
        print(p.map(square, numbers()))
    print(number)



class C():
    def __init__(self):
        self.cnt = 0

    def add(self, num):
        global number
        self.cnt += num
        number += num

    def add_cnt(self):
        with Pool(3) as p:   
            p.map(self.add, [1,2,3])


def time_consuming_func_split(num):
    # for t in range(10000000):
    #     num *= 12345
    #     num /= 12345
    # return num
    # print("Sleeping 1 sec...")
    # time.sleep(1)
    for t in range(5000000):
        num *= 12345
        num /= 12345
    print(num+10)


def do_sth(num):
    print("Sleeping 1 sec...")
    time.sleep(1)
    print('Done sleeping')
    print(num+10)


if __name__ == '__main__':
    s = time.time()

    # p1 = Process(target=time_consuming_func_split, args=(3, 0, 10000000))
    # p2 = Process(target=time_consuming_func_split, args=(3, 10000000, 20000000))
    # p3 = Process(target=time_consuming_func_split, args=(3, 20000000, 30000000))


    # p1.start()
    # p2.start()
    # p3.start()

    # p1.join()
    # p2.join()
    # p3.join()
    # e = time.time()
    # print(e-s)


    ps = []
    for i in range(8):
        p1 = Process(target=time_consuming_func_split, args=(i,))
        ps.append(p1)
        p1.start()


    for i in ps:
        i.join()
    



    e = time.time()
    print("Finished in", e-s)


    # Seq
    s = time.time()
    for i in range(8):
        time_consuming_func_split(i)


    e = time.time()
    print("Finished in", e-s)


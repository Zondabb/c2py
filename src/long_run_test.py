import threading
import time
import Extest

def test():
    Extest.long_running_test('a')

t = threading.Thread(target=test)
t.setDaemon(True)
t.start()

# t1 = threading.Thread(target=test)
# t1.setDaemon(True)
# t1.start()

Extest.long_running_test('a')
time.sleep(100)
# coding:utf-8
from subprocess import Popen
import os

def start_tensorboard():
    print "killing tensorboard............."
    os.system("""ps -ef | grep tensorboard | grep "tmp/tensorflow" | awk '{print $2}' | xargs kill -9""")

    print "starting tensorboard............"
    Popen("tensorboard  --logdir=/tmp/tensorflow/", shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)

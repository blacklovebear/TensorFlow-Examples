# coding:utf-8
from subprocess import Popen
import os

COMMON_PATH="/tmp/tensorflow/"

def start_tensorboard():
    print "killing tensorboard............."
    os.system("""ps -ef | grep tensorboard | grep "%s" | awk '{print $2}' | xargs kill -9""" % COMMON_PATH)

    print "starting tensorboard............"
    Popen("nohup tensorboard --logdir=%s > /tmp/tensorboard.log 2>&1 &" % COMMON_PATH,
        shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)


def model_dir(file_path, suffix):
    file_name = os.path.basename(file_path).split('.')[0]

    model_dir = COMMON_PATH + file_name + "/%s" % suffix
    return model_dir
#!/usr/bin/python
# Copyright (c) 2016 Baidu, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" module for launching cluster job """

import os
import argparse
import socket
import copy
import time
import signal

from fabric.api import run, put, settings, env, prefix
from fabric.tasks import execute

#configuration for cluster
import conf


def refine_unknown_args(cmd_args):
    '''
    refine unknown parameters to handle some special parameters
    '''
    new_args = []
    for arg in cmd_args:
        if arg.startswith("--") and arg.find("=") != -1:
            equal_pos = arg.find("=")  #find first = pos
            arglist = list(arg)
            arglist[equal_pos] = " "
            arg = "".join(arglist)
            arg = arg.lstrip("-")
            new_args += arg.split(" ")
        elif arg.startswith("--") and arg.find("=") == -1:
            arg = arg.lstrip("-")
            new_args.append(arg)
        else:
            new_args.append(arg)
    return new_args


def kill_process():
    '''
    kill comments threads
    '''
    run("ps aux \
         | grep paddle_process_by_paddle \
         | grep -v grep  \
         | awk '{print $2}' \
         | xargs kill > /dev/null 2>&1")


def job_prepare(topology, jobdir, logdir, modeldir):
    '''
    prepare job related workspace

    Assuming you already installed PaddlePaddle in all nodes which means
    PaddlePaddle related bins and dependencies libraries.
    Assuming the train/test data have already been installed.
    This function just prepare logdir and copy config file.
    '''

    def job_create_workspace():
        run('rm ' + logdir + ' -fr && ' + 'mkdir -p ' + logdir + ' ' + modeldir)
        put(config_file, modeldir) # save the topology
    def set_nodefile(nodeid):
        run('echo ' + str(nodeid) + ' > ' + logdir + '/nodefile')

    config_file = jobdir + "/" + topology + ".py"
    assert os.path.isfile(config_file)
    execute(job_create_workspace, hosts=conf.HOSTS)
    for i in xrange(len(conf.HOSTS)):
        execute(set_nodefile, i, hosts=conf.HOSTS[i])
    #clean rubbish caused by exception 
    with settings(warn_only=True):
        execute(kill_process, hosts=conf.HOSTS)


def job_pserver(jobdir, logdir, pids=None):
    '''
    start all pservers
    '''
    pargs = " --num_gradient_servers=" + str(len(conf.HOSTS))
    pargs += (" --nics=" + conf.PADDLE_NIC)
    pargs += " --port=" + str(conf.PADDLE_PORT)
    pargs += " --ports_num=" + str(conf.PADDLE_PORTS_NUM)
    #always start sparse pserver by default
    pargs += " --ports_num_for_sparse=" + str(conf.PADDLE_PORTS_NUM_FOR_SPARSE)
    pargs += " --comment=" + "paddle_process_by_paddle"

    def start_pserver(jobdir, pargs):
        '''
        start pserver process with fabric executor
        '''
        with prefix('export LD_LIBRARY_PATH=' + \
                conf.LD_LIBRARY_PATH + \
                ':$LD_LIBRARY_PATH'):
            program = 'paddle pserver'
            run('cd ' + jobdir + '; '  + \
                'GLOG_logtostderr=0 ' + \
                'GLOG_log_dir=' + logdir + \
                ' nohup ' + program + " " + pargs + \
                ' > ' + logdir + '/server.log 2>&1 < /dev/null & ',
                pty=False)

    execute(start_pserver, jobdir, pargs, hosts=conf.HOSTS)


def job_trainer(topology, jobdir, logdir, topo_ver, train_args_dict, modeldir=None, pids=None):
    '''
    start paddle trainer
    '''
    args = " --num_gradient_servers=" + str(len(conf.HOSTS))
    args += " --nics=" + conf.PADDLE_NIC
    args += " --port=" + str(conf.PADDLE_PORT)
    args += " --ports_num=" + str(conf.PADDLE_PORTS_NUM)
    args += " --comment=" + "paddle_process_by_paddle"
    ip_string = ""
    for i in xrange(len(conf.HOSTS)):
        host = conf.HOSTS[i]
        left = host.find("@")
        right = host.find(':')
        left = 0 if left == -1 else left + 1
        right = len(host) if right == -1 else right
        ip_string += (socket.gethostbyname(host[left:right]) + ",")
    ip_string = ip_string.rstrip(",")
    args += " --pservers=" + ip_string

    args_ext = ""
    for key, value in train_args_dict.items():
        args_ext += (' --' + key + '=' + value)

    args += " " + args_ext
    args += " --save_dir=" + modeldir
    args += " --config=" + str(topology) + ".py"
    args += " --config_args=batch_size=" + str(conf.BATCH_SIZE) + \
            ",use_mkldnn=" + str(conf.PADDLE_USE_MKLDNN) + \
            ",use_mkldnn_wgt=" + str(conf.PADDLE_USE_MKLDNN_WGT) + \
            ",use_dummy=" + str(conf.PADDLE_USE_DUMMY)
    if topology != "alexnet" and topo_ver is not None:
        if topology == "googlenet":
            args += ",version=" + str(topo_ver)
        else:
            args += ",layer_num=" + str(topo_ver)

    def start_trainer(args):
        '''
        start trainer process with fabric executor
        '''
        with prefix('export LD_LIBRARY_PATH=' + \
                conf.LD_LIBRARY_PATH + \
                ':$LD_LIBRARY_PATH'):
            program = 'paddle train'
            run('cd ' + jobdir + '; '  + \
                'GLOG_logtostderr=0 '
                'GLOG_log_dir=' + logdir + \
                ' nohup ' + program + " " + args + \
                ' > ' + logdir + '/train.log 2>&1 < /dev/null & ',
                pty=False)
    for i in xrange(len(conf.HOSTS)):
        train_args = copy.deepcopy(args)
        train_args += " --trainer_id=" + str(i)
        execute(start_trainer, train_args, hosts=conf.HOSTS[i])

def job_clean():
    '''
    if starting job failed from paddle internal, the framework always
    is launched successfully since these process are daemon processes.
    so this job_clean can alway clean job rubbish process with ctrl+c.
    '''

    def signal_handler(signal, frame):
        '''
        SIGINT handler
        '''

        def kill_process():
            run("ps aux \
                  | grep paddle_process_by_paddle \
                  | grep -v grep  \
                  | awk '{print $2}' \
                  | xargs kill > /dev/null 2>&1")

        with settings(warn_only=True):
            execute(kill_process, hosts=conf.HOSTS)

    signal.signal(signal.SIGINT, signal_handler)
    signal.pause()


def job_all(topology, jobdir, logdir, train_args_dict=None, topo_ver=None):
    modeldir = logdir + "/models"
    job_prepare(topology, jobdir, logdir, modeldir)
    job_pserver(jobdir, logdir)
    time.sleep(10)  #wait until pservers completely start
    job_trainer(topology, jobdir, logdir, topo_ver, train_args_dict, modeldir)
    job_clean()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="paddle.py", description='simple tool for cluster training')
    parser.add_argument(
        '-t',
        '--topology',
        required=True,
        default=None,
        help='topology name')
    parser.add_argument(
        '-l',
        '--topology_version',
        required=False,
        default=None,
        help='version or layer num for topology, none for alexnet')

    args, train_args_list = parser.parse_known_args()
    train_args = refine_unknown_args(train_args_list)
    train_args_dict = dict(zip(train_args[:-1:2], train_args[1::2]))
    assert args.topology in ['alexnet', 'googlenet', 'vgg', 'resnet']
    
    jobdir = os.path.abspath(conf.JOB_DIR)
    logdir = os.path.abspath(conf.LOG_DIR)
    assert os.path.isdir(jobdir)
    assert os.path.isdir(logdir)
    logdir += "/LOG_" + args.topology
    timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
    if args.topology == "alexnet":
        logdir += "_" + timestamp
        job_all(args.topology, jobdir, logdir, train_args_dict)
    else:
        version = args.topology_version
        if version == '' or version is None:
            if args.topology == "googlenet":
                version = "v1"
            elif args.topology == "vgg":
                version = 19
            elif args.topology == "resnet":
                version = 50
        else:
            if args.topology == "googlenet":
                assert version in ['v1'] #, 'v2'] only support v1 yet
            elif args.topology == "vgg":
                assert version in ['16', '19']
            elif args.topology == "resnet":
                assert version in ['50', '101', '152']
        logdir += "_" + str(version)
        logdir += "_" + timestamp    
        job_all(args.topology, jobdir, logdir, train_args_dict, version)
    
        

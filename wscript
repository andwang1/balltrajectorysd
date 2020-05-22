#! /usr/bin/env python
import os.path
import sys

# Adding Module paths (to take additional waf_tools from subdirs into account)
MODULES_PATH = os.path.abspath(os.path.join(sys.path[0], os.pardir, os.pardir, 'modules'))
for specific_module_folder in os.listdir(MODULES_PATH):
    sys.path.append(os.path.join(MODULES_PATH, specific_module_folder))
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], os.pardir)))

from waflib.Configure import conf

import sferes

PROJECT_NAME = "example-pytorch-sferes"


def get_relative_path(waf_tool_name):
    return PROJECT_NAME + '.' + 'waf_tools' + '.' + waf_tool_name


def options(opt):
    pass


@conf
def configure(conf):
    pass


def build(bld):
    bld.env.LIBPATH_PYTORCH = '/workspace/lib/torch/'
    bld.env.LIB_PYTORCH = 'torch c10 c10_cuda caffe2_detectron_ops_gpu caffe2_module_test_dynamic caffe2_nvrtc'.split(
        ' ')
    bld.env.INCLUDES_PYTORCH = ['/workspace/include/torch', '/workspace/include/torch/torch/csrc/api/include']

    bld.env.LIBPATH_PYTHON = '/usr/lib/x86_64-linux-gnu/'
    bld.env.LIB_PYTHON = ['python3.6m']
    bld.env.INCLUDES_PYTHON = '/usr/include/python3.6m'

    bld.env.LIBPATH_LIBFASTSIM = ['/workspace/lib']
    bld.env.LIB_LIBFASTSIM = ['fastsim']
    bld.env.INCLUDES_LIBFASTSIM = ['/workspace/include']

    bld.program(features='cxx',
            source='src/example_pytorch_libfastsim.cpp',
                includes='./src . ../../',
                uselib='TBB BOOST EIGEN PTHREAD MPI'
                       + ' PYTHON PYTORCH LIBFASTSIM SDL',
                use='sferes2',
                target='example_pytorch_libfastsim')



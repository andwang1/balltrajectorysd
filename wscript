#! /usr/bin/env python
import os.path
import sys
import sferes

print(sys.path[0])

# # Adding Module paths (to take additional waf_tools from subdirs into account)
# MODULES_PATH = os.path.abspath(os.path.join(sys.path[0], os.pardir, os.pardir, 'modules'))
# for specific_module_folder in os.listdir(MODULES_PATH):
#     sys.path.append(os.path.join(MODULES_PATH, specific_module_folder))
# sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], os.pardir)))

from waflib.Configure import conf

import sferes

print(sys.path[0])
sys.path.insert(0, sys.path[0]+'/waf_tools')
import boost
import eigen
import corrade
import magnum
import magnum_integration
import magnum_plugins

PROJECT_NAME = "balltrajectorysd"


def get_relative_path(waf_tool_name):
    return PROJECT_NAME + '.' + 'waf_tools' + '.' + waf_tool_name


def options(opt):
    opt.load('corrade')
    opt.load('magnum')
    opt.load('magnum_integration')
    opt.load('magnum_plugins')
    opt.load('robox2d')


# @conf
def configure(conf):
    print('conf exp:')
    conf.load('corrade')
    conf.load('magnum')
    conf.load('magnum_integration')
    conf.load('magnum_plugins')
    conf.load('robox2d')

    conf.check_corrade(components='Utility PluginManager', required=False)
    conf.env['magnum_dep_libs'] = 'MeshTools Primitives Shaders SceneGraph GlfwApplication'
    if conf.env['DEST_OS'] == 'darwin':
        conf.env['magnum_dep_libs'] += ' WindowlessCglApplication'
    else:
        conf.env['magnum_dep_libs'] += ' WindowlessGlxApplication'
    conf.check_magnum(components=conf.env['magnum_dep_libs'], required=False)
    conf.check_magnum_plugins(components='AssimpImporter', required=False)
    
    conf.get_env()['BUILD_MAGNUM'] = True
    conf.env['magnum_libs'] = magnum.get_magnum_dependency_libs(conf, conf.env['magnum_dep_libs'])
    conf.check_robox2d()
    
    print('done')


def build(bld):
    bld.env.LIBPATH_PYTORCH = '/workspace/lib/torch/'
    bld.env.LIB_PYTORCH = 'torch_cpu torch_cuda torch_python torch_global_deps shm caffe2_observers torch c10 c10_cuda caffe2_detectron_ops_gpu caffe2_module_test_dynamic caffe2_nvrtc'.split(' ')
    bld.env.INCLUDES_PYTORCH = ['/workspace/include/torch', '/workspace/include/torch/torch/csrc/api/include']

    bld.env.LIBPATH_PYTHON = '/usr/lib/x86_64-linux-gnu/'
    bld.env.LIB_PYTHON = ['python3.6m']
    bld.env.INCLUDES_PYTHON = '/usr/include/python3.6m'

    bld.env.INCLUDES_KDTREE = ['/workspace/include']

    print(bld.env['magnum_libs'])
    sferes.create_variants(bld,
                           source = 'src/balltrajectorysd.cpp',
                           includes='./src . ../../',
                           uselib='TBB BOOST EIGEN PTHREAD MPI'
                                + ' PYTHON PYTORCH KDTREE SDL ROBOX2D BOX2D' + bld.env['magnum_libs'],
                           use = 'sferes2',
                           target = 'balltrajectorysd',
                           variants = ['AE', 'VAE', 'AURORA'])
#!/bin/bash
./waf configure --exp example-pytorch-sferes --cpp14=yes --kdtree /workspace/include

./waf --exp example-pytorch-sferes -j 1
echo 'FINISHED BUILDING. Now fixing name of files'
python -m fix_build --path-folder /git/sferes2/build/exp/example-pytorch-sferes/

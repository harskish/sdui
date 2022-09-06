<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# BUILDING

Install dependencies
```
conda create --name tvm python=3.10
conda activate tvm
conda install -c apple tensorflow-deps -y && conda install -y numpy decorator attrs cython llvmdev tornado psutil xgboost cloudpickle pytest
pip install delocate
```

Clone TVM

```
git clone --recursive https://github.com/apache/tvm tvm
cd tvm && mkdir build
cp cmake/config.cmake build
```

Edit the `config.cmake` file in the build directory setting the following

```
USE_METAL ON
USE_LLVM ON
USE_OPENMP gnu
USE_BNNS ON
USE_RELAY_DEBUG ON
```

Build TVM

```
cd build
cmake -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_BUILD_TYPE=Debug ..  # Debug, Release, RelWithDebInfo, MinSizeRel
cmake --build .
cd ../python
python setup.py bdist_wheel  # will use dylibs compiled by cmake
cd dist
delocate-wheel -w fixed -v tvm-0.9.0-cp310-cp310-macosx_12_0_arm64.whl
```

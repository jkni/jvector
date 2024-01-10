#!/bin/bash

# Copyright DataStax, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

gcc -fPIC -O3 -march=native -shared -o libjvector.so jvector_simd.c

# Generate Java source code
# Should only be run when c header changes
jextract --source \
  --output ../java \
  -t io.github.jbellis.jvector.vector.cnative \
  -I . \
  -l jvector \
  --header-class-name NativeSimdOps \
  jvector_simd.h
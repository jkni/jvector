/*
 * Copyright DataStax, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.github.jbellis.jvector.vector.types;

import java.io.DataOutput;
import java.io.IOException;

import io.github.jbellis.jvector.disk.RandomAccessReader;

public interface VectorTypeSupport {
    VectorFloat<?> createFloatType(Object data);
    VectorFloat<?> createFloatType(int length);

    VectorFloat<?> readFloatType(RandomAccessReader r, int size) throws IOException;
    void writeFloatType(DataOutput out, VectorFloat<?> vector) throws IOException;

    VectorByte<?> createByteType(Object data);
    VectorByte<?> createByteType(int length);

    VectorByte<?> readByteType(RandomAccessReader r, int size) throws IOException;

    void readByteType(RandomAccessReader r, VectorByte<?> vector) throws IOException;

    void writeByteType(DataOutput out, VectorByte<?> vector) throws IOException;
}

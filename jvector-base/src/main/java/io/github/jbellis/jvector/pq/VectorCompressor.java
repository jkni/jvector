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

package io.github.jbellis.jvector.pq;

import java.io.DataOutput;
import java.io.IOException;
import java.util.List;
import java.util.Map;

/**
 * Interface for vector compression.  T is the encoded (compressed) vector type;
 * it will be an array type.
 */
public interface VectorCompressor<T> {
    T[] encodeAll(List<float[]> vectors);
    default T[] encodeAll(List<float[]> vectors, Map<Integer, Integer> renumbering) {
        throw new UnsupportedOperationException();
    }

    T encode(float[] v);

    void write(DataOutput out) throws IOException;

    /**
     * @param compressedVectors must match the type T for this VectorCompressor, but
     *                          it is declared as Object because we want callers to be able to use this
     *                          without committing to a specific type T.
     */
    CompressedVectors createCompressedVectors(Object[] compressedVectors);
}

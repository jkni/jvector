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

import io.github.jbellis.jvector.graph.NodeSimilarity;
import io.github.jbellis.jvector.util.Accountable;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;

import java.io.DataOutput;
import java.io.IOException;

public interface CompressedVectors extends Accountable {
    /** write the compressed vectors to the given DataOutput */
    void write(DataOutput out) throws IOException;

    default NodeSimilarity.ApproximateScoreFunction approximateScoreFunctionFor(float[] q, VectorSimilarityFunction similarityFunction, boolean precompute) {
        return approximateScoreFunctionFor(q, similarityFunction);
    }

    /** @return an approximate score function for the given query vector and similarity function, defaulting to precomputation */
    NodeSimilarity.ApproximateScoreFunction approximateScoreFunctionFor(float[] q, VectorSimilarityFunction similarityFunction);

    /** @return the original size of the vectors, in bytes, before compression */
    int getOriginalSize();

    /** @return the compressed size of the vectors, in bytes */
    int getCompressedSize();

    /** @return the compressor used by this instance */
    VectorCompressor<?> getCompressor();
}

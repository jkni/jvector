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
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorByte;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

/**
 * Performs similarity comparisons with compressed vectors without decoding them.
 * These decoders use Quick(er) ADC-style transposed vectors fused into a graph.
 */
public abstract class QuickADCPQDecoder implements NodeSimilarity.ApproximateScoreFunction {
    protected final PQVectors pqv;
    protected static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();

    protected QuickADCPQDecoder(PQVectors pqv) {
        this.pqv = pqv;
    }

    protected static abstract class CachingDecoder extends QuickADCPQDecoder {
        protected final VectorFloat<?> partialSums;
        protected CachingDecoder(PQVectors pqv, VectorFloat<?> query, VectorSimilarityFunction vsf) {
            super(pqv);
            partialSums = pqv.reusablePartialSums();
            var pq = this.pqv.pq;

            VectorFloat<?> center = pq.getCenter();
            var centeredQuery = center == null ? query : VectorUtil.sub(query, center);
            var step = 5f / (127 * pq.getSubspaceCount());
            //int[] indexCounts = new int[128];
            for (var i = 0; i < pq.getSubspaceCount(); i++) {
                int offset = pq.subvectorSizesAndOffsets[i][1];
                int baseOffset = i * pq.getClusterCount();
                for (var j = 0; j < pq.getClusterCount(); j++) {
                    VectorFloat<?> centroidSubvector = pq.codebooks[i][j];
                    switch (vsf) {
                        case DOT_PRODUCT:
                            partialSums.set(baseOffset + j, VectorUtil.dotProduct(centroidSubvector, 0, centeredQuery, offset, centroidSubvector.length()));
                            break;
                        case EUCLIDEAN:
                            partialSums.set(baseOffset + j, VectorUtil.squareDistance(centroidSubvector, 0, centeredQuery, offset, centroidSubvector.length()));
                            break;
                        default:
                            throw new UnsupportedOperationException("Unsupported similarity function " + vsf);
                    }
                }
            }
        }

        protected float decodedSimilarity(VectorByte<?> encoded) {
            return VectorUtil.assembleAndSum(partialSums, pqv.pq.getClusterCount(), encoded);
        }
    }

     static class DotProductDecoder extends CachingDecoder {
        private final OnDiskADCGraphIndex<VectorFloat<?>> fgi;
        private final OnDiskADCGraphIndex<VectorFloat<?>>.OnDiskView fgiView;
        private final VectorFloat<?> results;

        public DotProductDecoder(OnDiskADCGraphIndex<VectorFloat<?>> fgi, VectorFloat<?> query) {
            super(fgi.pqv, query, VectorSimilarityFunction.DOT_PRODUCT);
            this.fgi = fgi;
            this.fgiView = fgi.scoreView.get();
            this.results = fgi.reusableResults();
        }

        @Override
        public float similarityTo(int node2) {
            return (1 + decodedSimilarity(pqv.get(node2))) / 2;
        }

        @Override
        public VectorFloat<?> bulkSimilarityTo(int origin) {
            var permutedNodes = fgiView.getPackedNeighbors(origin);
            results.zero();
            VectorUtil.bulkShuffleSimilarity(permutedNodes, fgi.pqv.getCompressedSize(), partialSums, results, VectorSimilarityFunction.DOT_PRODUCT);
            return results;
        }

        @Override
        public boolean supportsBulkSimilarity() {
            return true;
        }
    }

    static class EuclideanDecoder extends CachingDecoder {
        private final OnDiskADCGraphIndex fgi;
        private final OnDiskADCGraphIndex<VectorFloat<?>>.OnDiskView fgiView;
        private final VectorFloat<?> results;

        public EuclideanDecoder(OnDiskADCGraphIndex<VectorFloat<?>> fgi, VectorFloat<?> query) {
            super(fgi.pqv, query, VectorSimilarityFunction.EUCLIDEAN);
            this.fgi = fgi;
            this.fgiView = fgi.scoreView.get();
            this.results = fgi.reusableResults();
        }

        @Override
        public float similarityTo(int node2) {
            return 1 / (1 + decodedSimilarity(pqv.get(node2)));
        }

        @Override
        public VectorFloat<?> bulkSimilarityTo(int origin) {
            var permutedNodes = fgiView.getPackedNeighbors(origin);
            results.zero();
            VectorUtil.bulkShuffleSimilarity(permutedNodes, fgi.pqv.getCompressedSize(), partialSums, results, VectorSimilarityFunction.EUCLIDEAN);
            return results;
        }

        @Override
        public boolean supportsBulkSimilarity() {
            return true;
        }
    }
}

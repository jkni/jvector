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

import io.github.jbellis.jvector.disk.CachingFusedGraphIndex;
import io.github.jbellis.jvector.disk.CachingGraphIndex;
import io.github.jbellis.jvector.disk.OnDiskFusedGraphIndex;
import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.NodeSimilarity;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;

/**
 * Performs similarity comparisons with compressed vectors without decoding them
 */
public abstract class FastPQDecoder implements NodeSimilarity.ApproximateScoreFunction {
    protected final PQVectors cv;

    protected FastPQDecoder(PQVectors cv) {
        this.cv = cv;
    }

    protected static abstract class CachingDecoder extends FastPQDecoder {
        protected final ThreadLocal<byte[]> partials = ThreadLocal.withInitial(() -> new byte[ProductQuantization.CLUSTERS * this.cv.pq.getSubspaceCount()]);
        protected final byte[] tlPartials;

        protected CachingDecoder(PQVectors cv, float[] query, VectorSimilarityFunction vsf) {
            super(cv);
            var pq = this.cv.pq;
            tlPartials = partials.get();

            float[] center = pq.getCenter();
            var centeredQuery = center == null ? query : VectorUtil.sub(query, center);
            var step = 2f / (127 * pq.getSubspaceCount());
            int[] indexCounts = new int[128];
            for (var i = 0; i < pq.getSubspaceCount(); i++) {
                int offset = pq.subvectorSizesAndOffsets[i][1];
                int baseOffset = i * ProductQuantization.CLUSTERS;
                for (var j = 0; j < ProductQuantization.CLUSTERS; j++) {
                    float[] centroidSubvector = pq.codebooks[i][j];
                    switch (vsf) {
                        case DOT_PRODUCT:
                            var dotProduct = VectorUtil.dotProduct(centroidSubvector, 0, centeredQuery, offset, centroidSubvector.length);
                            var dotProductShifted = dotProduct + 1/pq.getSubspaceCount();
                            // dotProductShifted divided by step, must fall in the range 0 - 127
                            var index = (byte) Math.min(127, Math.max(0, Math.round(dotProductShifted / step)));
                            indexCounts[index] = indexCounts[index] + 1;
                            tlPartials[baseOffset + j] = index;
                            break;
                        case EUCLIDEAN:
                            throw new UnsupportedOperationException("Unsupported similarity function " + vsf);
                        default:
                            throw new UnsupportedOperationException("Unsupported similarity function " + vsf);
                    }
                }
            }
        }

        protected float decodedSimilarity(byte[] encoded) {
            return VectorUtil.shuffleAndSum(tlPartials, ProductQuantization.CLUSTERS, encoded);
        }
    }

    public static class DotProductDecoder extends CachingDecoder {
        private final GraphIndex.View<float[]> fgi;
        public DotProductDecoder(PQVectors cv, CachingFusedGraphIndex fgi, float[] query) {
            super(cv, query, VectorSimilarityFunction.DOT_PRODUCT);
            this.fgi = fgi.getView();
        }

        @Override
        public float similarityTo(int node2) {
            return (1 + decodedSimilarity(cv.get(node2))) / 2;
        }

        @Override
        public float[] bulkSimilarityTo(int node2) {
            // look up nodes in cv, but arrange them to be put the 0th component of all vectors, then 1st component of all vectors, etc
            var permutedNodes = fgi.getPackedNeighbors(node2);
            return VectorUtil.bulkShuffleSimilarity(permutedNodes, cv.getCompressedSize(), tlPartials, 0L);
        }
    }

    static class EuclideanDecoder extends CachingDecoder {
        public EuclideanDecoder(PQVectors cv, float[] query) {
            super(cv, query, VectorSimilarityFunction.EUCLIDEAN);
        }

        @Override
        public float similarityTo(int node2) {
            return 1 / (1 + decodedSimilarity(cv.get(node2)));
        }
    }

    static class CosineDecoder extends FastPQDecoder {
        protected final float[] partialSums;
        protected final float[] aMagnitude;
        protected final float bMagnitude;

        public CosineDecoder(PQVectors cv, float[] query) {
            super(cv);
            var pq = this.cv.pq;

            // Compute and cache partial sums and magnitudes for query vector
            partialSums = cv.reusablePartialSums();
            aMagnitude = cv.reusablePartialMagnitudes();
            float bMagSum = 0.0f;

            float[] center = pq.getCenter();
            float[] centeredQuery = center == null ? query : VectorUtil.sub(query, center);

            for (int m = 0; m < pq.getSubspaceCount(); ++m) {
                int offset = pq.subvectorSizesAndOffsets[m][1];
                for (int j = 0; j < ProductQuantization.CLUSTERS; ++j) {
                    float[] centroidSubvector = pq.codebooks[m][j];
                    partialSums[(m * ProductQuantization.CLUSTERS) + j] = VectorUtil.dotProduct(centroidSubvector, 0, centeredQuery, offset, centroidSubvector.length);
                    aMagnitude[(m * ProductQuantization.CLUSTERS) + j] = VectorUtil.dotProduct(centroidSubvector, 0, centroidSubvector, 0, centroidSubvector.length);
                }

                bMagSum += VectorUtil.dotProduct(centeredQuery, offset, centeredQuery, offset, pq.subvectorSizesAndOffsets[m][0]);
            }

            this.bMagnitude = bMagSum;
        }

        @Override
        public float similarityTo(int node2) {
            return (1 + decodedCosine(node2)) / 2;
        }

        protected float decodedCosine(int node2) {
            float sum = 0.0f;
            float aMag = 0.0f;

            byte[] encoded = cv.get(node2);

            for (int m = 0; m < encoded.length; ++m) {
                int centroidIndex = Byte.toUnsignedInt(encoded[m]);
                sum += partialSums[(m * ProductQuantization.CLUSTERS) + centroidIndex];
                aMag += aMagnitude[(m * ProductQuantization.CLUSTERS) + centroidIndex];
            }

            return (float) (sum / Math.sqrt(aMag * bMagnitude));
        }
    }
}

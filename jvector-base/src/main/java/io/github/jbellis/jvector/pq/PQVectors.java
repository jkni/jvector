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

import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.graph.NodeSimilarity;
import io.github.jbellis.jvector.util.RamUsageEstimator;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;

import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;
import java.util.Objects;

public class PQVectors implements CompressedVectors {
    final ProductQuantization pq;
    private final byte[][] compressedVectors;
    private final ThreadLocal<float[]> partialSums; // for dot product, euclidean, and cosine
    private final ThreadLocal<float[]> partialMagnitudes; // for cosine
    private final ThreadLocal<float[]> scratch;

    public PQVectors(ProductQuantization pq, byte[][] compressedVectors)
    {
        this.pq = pq;
        this.compressedVectors = compressedVectors;
        this.partialSums = ThreadLocal.withInitial(() -> new float[pq.getSubspaceCount() * ProductQuantization.CLUSTERS]);
        this.partialMagnitudes = ThreadLocal.withInitial(() -> new float[pq.getSubspaceCount() * ProductQuantization.CLUSTERS]);
        this.scratch = ThreadLocal.withInitial(() -> new float[pq.getSubspaceCount()]);
    }

    @Override
    public void write(DataOutput out) throws IOException
    {
        // pq codebooks
        pq.write(out);

        // compressed vectors
        out.writeInt(compressedVectors.length);
        out.writeInt(pq.getSubspaceCount());
        for (var v : compressedVectors) {
            out.write(v);
        }
    }

    public static CompressedVectors load(RandomAccessReader in, long offset) throws IOException
    {
        in.seek(offset);

        // pq codebooks
        var pq = ProductQuantization.load(in);

        // read the vectors
        int size = in.readInt();
        if (size < 0) {
            throw new IOException("Invalid compressed vector count " + size);
        }
        var compressedVectors = new byte[size][];

        int compressedDimension = in.readInt();
        if (compressedDimension < 0) {
            throw new IOException("Invalid compressed vector dimension " + compressedDimension);
        }

        for (int i = 0; i < size; i++)
        {
            byte[] vector = new byte[compressedDimension];
            in.readFully(vector);
            compressedVectors[i] = vector;
        }

        return new PQVectors(pq, compressedVectors);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        PQVectors that = (PQVectors) o;
        if (!Objects.equals(pq, that.pq)) return false;
        if (compressedVectors.length != that.compressedVectors.length) return false;
        return Arrays.deepEquals(compressedVectors, that.compressedVectors);
    }

    @Override
    public int hashCode() {
        return Objects.hash(pq, Arrays.deepHashCode(compressedVectors));
    }

    /**
     * @param q the query vector
     * @param similarityFunction the similarity function to use
     * @param precompute whether to precompute score function fragments. This is provided for situations when too few
     *                   vectors are being scored to make precomputation worthwhile.
     * @return a ScoreFunction suitable for performing search against the compressed vectors,
     * potentially without decompression them first
     */
    @Override
    public NodeSimilarity.ApproximateScoreFunction approximateScoreFunctionFor(float[] q, VectorSimilarityFunction similarityFunction, boolean precompute) {
        if (precompute) {
            return approximateScoreFunctionFor(q, similarityFunction);
        } else {
            var center = pq.getCenter();
            var centeredQuery = center == null ? q : VectorUtil.sub(q, center);
            switch (similarityFunction) {
                case DOT_PRODUCT:
                    return i -> (1 + decodedDotProduct(compressedVectors[i], centeredQuery)) / 2;
                case EUCLIDEAN:
                    return i -> 1 / (1 + decodedSquareDistance(compressedVectors[i], centeredQuery));
                case COSINE:
                    return i -> (1 + decodedCosine(compressedVectors[i], centeredQuery)) / 2;
                default:
                    throw new IllegalArgumentException("Unsupported similarity function " + similarityFunction);
            }
        }
    }

    /**
     * Computes the dot product of the (approximate) original decoded vector with
     * another vector.
     * <p>
     * This method can compute the dot product without materializing the decoded vector as a new float[],
     * which will be roughly 2x as fast as decode() + dot().
     * <p>
     * It is the caller's responsibility to center the `other` vector by subtracting the global centroid
     * before calling this method.
     */
    private float decodedDotProduct(byte[] encoded, float[] other) {
        var a = scratch.get();
        for (int m = 0; m < pq.getSubspaceCount(); ++m) {
            int offset = pq.subvectorSizesAndOffsets[m][1];
            int centroidIndex = Byte.toUnsignedInt(encoded[m]);
            float[] centroidSubvector = pq.codebooks[m][centroidIndex];
            a[m] = VectorUtil.dotProduct(centroidSubvector, 0, other, offset, centroidSubvector.length);
        }
        return VectorUtil.sum(a);
    }

    /**
     * Computes the square distance of the (approximate) original decoded vector with
     * another vector.
     * <p>
     * This method can compute the square distance without materializing the decoded vector as a new float[],
     * which will be roughly 2x as fast as decode() + squaredistance().
     * <p>
     * It is the caller's responsibility to center the `other` vector by subtracting the global centroid
     * before calling this method.
     */
    private float decodedSquareDistance(byte[] encoded, float[] other) {
        float sum = 0.0f;
        var a = scratch.get();
        for (int m = 0; m < pq.getSubspaceCount(); ++m) {
            int offset = pq.subvectorSizesAndOffsets[m][1];
            int centroidIndex = Byte.toUnsignedInt(encoded[m]);
            float[] centroidSubvector = pq.codebooks[m][centroidIndex];
            a[m] = VectorUtil.squareDistance(centroidSubvector, 0, other, offset, centroidSubvector.length);
        }
        return VectorUtil.sum(a);
    }

    /**
     * Computes the cosine of the (approximate) original decoded vector with
     * another vector.
     * <p>
     * This method can compute the cosine without materializing the decoded vector as a new float[],
     * which will be roughly 1.5x as fast as decode() + dot().
     * <p>
     * It is the caller's responsibility to center the `other` vector by subtracting the global centroid
     * before calling this method.
     */
    private float decodedCosine(byte[] encoded, float[] other) {
        float sum = 0.0f;
        float aMagnitude = 0.0f;
        float bMagnitude = 0.0f;
        for (int m = 0; m < pq.getSubspaceCount(); ++m) {
            int offset = pq.subvectorSizesAndOffsets[m][1];
            int centroidIndex = Byte.toUnsignedInt(encoded[m]);
            float[] centroidSubvector = pq.codebooks[m][centroidIndex];
            var length = centroidSubvector.length;
            sum += VectorUtil.dotProduct(centroidSubvector, 0, other, offset, length);
            aMagnitude += VectorUtil.dotProduct(centroidSubvector, 0, centroidSubvector, 0, length);
            bMagnitude +=  VectorUtil.dotProduct(other, offset, other, offset, length);
        }
        return (float) (sum / Math.sqrt(aMagnitude * bMagnitude));
    }

    @Override
    public NodeSimilarity.ApproximateScoreFunction approximateScoreFunctionFor(float[] q, VectorSimilarityFunction similarityFunction) {
        switch (similarityFunction) {
            case DOT_PRODUCT:
                return new PQDecoder.DotProductDecoder(this, q);
            case EUCLIDEAN:
                return new PQDecoder.EuclideanDecoder(this, q);
            case COSINE:
                return new PQDecoder.CosineDecoder(this, q);
            default:
                throw new IllegalArgumentException("Unsupported similarity function " + similarityFunction);
        }
    }


    byte[] get(int ordinal) {
        return compressedVectors[ordinal];
    }

    float[] reusablePartialSums() {
        return partialSums.get();
    }

    float[] reusablePartialMagnitudes() {
        return partialMagnitudes.get();
    }

    @Override
    public int getOriginalSize() {
        return pq.originalDimension * Float.BYTES;
    }

    @Override
    public int getCompressedSize() {
        return pq.codebooks.length;
    }

    @Override
    public ProductQuantization getCompressor() {
        return pq;
    }

    @Override
    public long ramBytesUsed() {
        long codebooksSize = pq.memorySize();
        long compressedVectorSize = RamUsageEstimator.sizeOf(compressedVectors[0]);
        return codebooksSize + (compressedVectorSize * compressedVectors.length);
    }
}

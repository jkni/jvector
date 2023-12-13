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

import io.github.jbellis.jvector.disk.Io;
import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.util.PoolingSupport;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static java.lang.Math.min;

/**
 * Binary Quantization of float vectors: each float is compressed to a single bit,
 * and similarity is computed with a simple Hamming distance.
 */
public class BinaryQuantization implements VectorCompressor<long[]> {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();
    private final VectorFloat<?> globalCentroid;

    public BinaryQuantization(VectorFloat<?> globalCentroid) {
        this.globalCentroid = globalCentroid;
    }

    public static BinaryQuantization compute(RandomAccessVectorValues<VectorFloat<?>> ravv) {
        return compute(ravv, ForkJoinPool.commonPool());
    }

    public static BinaryQuantization compute(RandomAccessVectorValues<VectorFloat<?>> ravv, ForkJoinPool parallelExecutor) {
        // limit the number of vectors we train on
        var P = min(1.0f, ProductQuantization.MAX_PQ_TRAINING_SET_SIZE / (float) ravv.size());
        var ravvCopy = ravv.isValueShared() ? PoolingSupport.newThreadBased(ravv::copy) : PoolingSupport.newNoPooling(ravv);
        var vectors = parallelExecutor.submit(() -> IntStream.range(0, ravv.size()).parallel()
                .filter(i -> ThreadLocalRandom.current().nextFloat() < P)
                .mapToObj(targetOrd -> {
                    try (var pooledRavv = ravvCopy.get()) {
                        var localRavv = pooledRavv.get();
                        VectorFloat<?> v = localRavv.vectorValue(targetOrd);
                        return localRavv.isValueShared() ? v.copy() : v;
                    }
                })
                .collect(Collectors.toList()))
                .join();

        // compute the centroid of the training set
        VectorFloat<?> globalCentroid = KMeansPlusPlusClusterer.centroidOf(vectors);
        return new BinaryQuantization(globalCentroid);
    }

    @Override
    public CompressedVectors createCompressedVectors(Object[] compressedVectors) {
        return new BQVectors(this, (long[][]) compressedVectors);
    }

    @Override
    public long[][] encodeAll(List<VectorFloat<?>> vectors, ForkJoinPool simdExecutor) {
        return simdExecutor.submit(() -> vectors.stream().parallel().map(this::encode).toArray(long[][]::new)).join();
    }

    /**
     * Encodes the input vector
     *
     * @return one bit per original f32
     */
    @Override
    public long[] encode(VectorFloat<?> v) {
        var centered = VectorUtil.sub(v, globalCentroid);

        int M = (int) Math.ceil(centered.length() / 64.0);
        long[] encoded = new long[M];
        for (int i = 0; i < M; i++) {
            long bits = 0;
            for (int j = 0; j < 64; j++) {
                int idx = i * 64 + j;
                if (idx >= centered.length()) {
                    break;
                }
                if (centered.get(idx) > 0) {
                    bits |= 1L << j;
                }
            }
            encoded[i] = bits;
        }
        return encoded;
    }

    @Override
    public void write(DataOutput out) throws IOException {
        out.writeInt(globalCentroid.length());
        vectorTypeSupport.writeFloatType(out, globalCentroid);
    }

    public int getOriginalDimension() {
        return globalCentroid.length();
    }

    public static BinaryQuantization load(RandomAccessReader in) throws IOException {
        int length = in.readInt();
        var centroid = vectorTypeSupport.readFloatType(in, length);
        return new BinaryQuantization(centroid);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        BinaryQuantization that = (BinaryQuantization) o;
        return Objects.equals(globalCentroid, that.globalCentroid);
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(globalCentroid);
    }

    @Override
    public String toString() {
        return "BinaryQuantization";
    }
}

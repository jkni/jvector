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
package io.github.jbellis.jvector.microbench;


import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.vector.NativeVectorizationProvider;
import io.github.jbellis.jvector.vector.PanamaVectorizationProvider;
import io.github.jbellis.jvector.vector.VectorUtilSupport;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.Threads;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

import java.util.Random;

@Warmup(iterations = 2, time = 5)
@Measurement(iterations = 20, time = 1)
@Fork(warmups = 1, value = 1, jvmArgsAppend = {"--add-modules=jdk.incubator.vector", "--enable-preview", "-Djvector.experimental.enable_native_vectorization=true"})
public class SimilarityBench {
    final static VectorUtilSupport nativeSupport = new NativeVectorizationProvider().getVectorUtilSupport();
    final static VectorUtilSupport panamaSupport = new PanamaVectorizationProvider().getVectorUtilSupport();
    final static VectorTypeSupport panamaTypeSupport = new PanamaVectorizationProvider().getVectorTypeSupport();

    static VectorFloat<?> A_4 = TestUtil.randomVector(new Random(), 4);
    static VectorFloat<?> B_4 = TestUtil.randomVector(new Random(), 4);
    static VectorFloat<?> A_8 = TestUtil.randomVector(new Random(), 8);
    static VectorFloat<?> B_8 = TestUtil.randomVector(new Random(), 8);
    static VectorFloat<?> A_16 = TestUtil.randomVector(new Random(), 16);
    static VectorFloat<?> B_16 = TestUtil.randomVector(new Random(), 16);
    static VectorFloat<?> A_32 = TestUtil.randomVector(new Random(), 32);
    static VectorFloat<?> B_32 = TestUtil.randomVector(new Random(), 32);
    static VectorFloat<?> A_64 = TestUtil.randomVector(new Random(), 64);
    static VectorFloat<?> B_64 = TestUtil.randomVector(new Random(), 64);
    static VectorFloat<?> A_1536 = TestUtil.randomVector(new Random(), 1536);
    static VectorFloat<?> B_1536 = TestUtil.randomVector(new Random(), 1536);
    static VectorFloat<?> A_12288 = TestUtil.randomVector(new Random(), 12288);
    static VectorFloat<?> B_12288 = TestUtil.randomVector(new Random(), 12288);
    /*static VectorFloat<?> A_ONHEAP_4 = panamaTypeSupport.createFloatVector(4);
    static VectorFloat<?> B_ONHEAP_4 = panamaTypeSupport.createFloatVector(4);
    static VectorFloat<?> A_ONHEAP_8 = panamaTypeSupport.createFloatVector(8);
    static VectorFloat<?> B_ONHEAP_8 = panamaTypeSupport.createFloatVector(8);
    static VectorFloat<?> A_ONHEAP_16 = panamaTypeSupport.createFloatVector(16);
    static VectorFloat<?> B_ONHEAP_16 = panamaTypeSupport.createFloatVector(16);
    static VectorFloat<?> A_ONHEAP_32 = panamaTypeSupport.createFloatVector(32);
    static VectorFloat<?> B_ONHEAP_32 = panamaTypeSupport.createFloatVector(32);
    static VectorFloat<?> A_ONHEAP_64 = panamaTypeSupport.createFloatVector(64);
    static VectorFloat<?> B_ONHEAP_64 = panamaTypeSupport.createFloatVector(64);
    static VectorFloat<?> A_ONHEAP_1536 = panamaTypeSupport.createFloatVector(1536);
    static VectorFloat<?> B_ONHEAP_1536 = panamaTypeSupport.createFloatVector(1536);
    static VectorFloat<?> A_ONHEAP_12288 = panamaTypeSupport.createFloatVector(12288);
    static VectorFloat<?> B_ONHEAP_12288 = panamaTypeSupport.createFloatVector(12288);

    static {
        for (int i = 0; i < A_4.length(); i++) {
            A_ONHEAP_4.set(i, A_4.get(i));
            B_ONHEAP_4.set(i, B_4.get(i));
        }
        for (int i = 0; i < A_8.length(); i++) {
            A_ONHEAP_8.set(i, A_8.get(i));
            B_ONHEAP_8.set(i, B_8.get(i));
        }
        for (int i = 0; i < A_16.length(); i++) {
            A_ONHEAP_16.set(i, A_16.get(i));
            B_ONHEAP_16.set(i, B_16.get(i));
        }
        for (int i = 0; i < A_32.length(); i++) {
            A_ONHEAP_32.set(i, A_32.get(i));
            B_ONHEAP_32.set(i, B_32.get(i));
        }
        for (int i = 0; i < A_64.length(); i++) {
            A_ONHEAP_64.set(i, A_64.get(i));
            B_ONHEAP_64.set(i, B_64.get(i));
        }
        for (int i = 0; i < A_1536.length(); i++) {
            A_ONHEAP_1536.set(i, A_1536.get(i));
            B_ONHEAP_1536.set(i, B_1536.get(i));
        }
        for (int i = 0; i < A_12288.length(); i++) {
            A_ONHEAP_12288.set(i, A_12288.get(i));
            B_ONHEAP_12288.set(i, B_12288.get(i));
        }
    }*/

    @Benchmark
    @BenchmarkMode(Mode.Throughput)
    @Threads(8)
    public void testDotProduct_4(Blackhole bh) {
        bh.consume(nativeSupport.dotProduct(A_4, B_4));
    }

    @Benchmark
    @BenchmarkMode(Mode.Throughput)
    @Threads(8)
    public void testDotProduct_8(Blackhole bh) {
        bh.consume(nativeSupport.dotProduct(A_8, B_8));
    }


    @Benchmark
    @BenchmarkMode(Mode.Throughput)
    @Threads(8)
    public void testDotProduct_16(Blackhole bh) {
        bh.consume(nativeSupport.dotProduct(A_16, B_16));
    }

    @Benchmark
    @BenchmarkMode(Mode.Throughput)
    @Threads(8)
    public void testDotProduct_32(Blackhole bh) {
        bh.consume(nativeSupport.dotProduct(A_32, B_32));
    }

    @Benchmark
    @BenchmarkMode(Mode.Throughput)
    @Threads(8)
    public void testDotProduct_64(Blackhole bh) {
        bh.consume(nativeSupport.dotProduct(A_64, B_64));
    }

    @Benchmark
    @BenchmarkMode(Mode.Throughput)
    @Threads(8)
    public void testDotProduct_1536(Blackhole bh) {
        bh.consume(nativeSupport.dotProduct(A_1536, B_1536));
    }

    @Benchmark
    @BenchmarkMode(Mode.Throughput)
    @Threads(8)
    public void testDotProduct_12288(Blackhole bh) {
        bh.consume(nativeSupport.dotProduct(A_12288, B_12288));
    }

    /*@Benchmark
    @BenchmarkMode(Mode.Throughput)
    @Threads(8)
    public void testDotProduct_Panama4(Blackhole bh) {
        bh.consume(panamaSupport.dotProduct(A_ONHEAP_4, B_ONHEAP_4));
    }

    @Benchmark
    @BenchmarkMode(Mode.Throughput)
    @Threads(8)
    public void testDotProduct_Panama8(Blackhole bh) {
        bh.consume(panamaSupport.dotProduct(A_ONHEAP_8, B_ONHEAP_8));
    }


    @Benchmark
    @BenchmarkMode(Mode.Throughput)
    @Threads(8)
    public void testDotProduct_Panama16(Blackhole bh) {
        bh.consume(panamaSupport.dotProduct(A_ONHEAP_16, B_ONHEAP_16));
    }

    @Benchmark
    @BenchmarkMode(Mode.Throughput)
    @Threads(8)
    public void testDotProduct_Panama32(Blackhole bh) {
        bh.consume(panamaSupport.dotProduct(A_ONHEAP_32, B_ONHEAP_32));
    }

    @Benchmark
    @BenchmarkMode(Mode.Throughput)
    @Threads(8)
    public void testDotProduct_Panama64(Blackhole bh) {
        bh.consume(panamaSupport.dotProduct(A_ONHEAP_64, B_ONHEAP_64));
    }

    @Benchmark
    @BenchmarkMode(Mode.Throughput)
    @Threads(8)
    public void testDotProduct_Panama1536(Blackhole bh) {
        bh.consume(panamaSupport.dotProduct(A_ONHEAP_1536, B_ONHEAP_1536));
    }

    @Benchmark
    @BenchmarkMode(Mode.Throughput)
    @Threads(8)
    public void testDotProduct_Panama12288(Blackhole bh) {
        bh.consume(panamaSupport.dotProduct(A_ONHEAP_12288, B_ONHEAP_12288));
    }*/


    public static void main(String[] args) throws Exception {
        org.openjdk.jmh.Main.main(args);
    }
}


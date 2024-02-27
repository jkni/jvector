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


import jdk.incubator.vector.FloatVector;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Threads;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;

@Warmup(iterations = 3, time = 5)
@Measurement(iterations = 20, time = 2)
@Fork(warmups = 1, value = 1, jvmArgsAppend = {"--add-modules=jdk.incubator.vector", "--enable-preview", "-Djvector.experimental.enable_native_vectorization=true"})
public class VectorLoadBench {
    @State(Scope.Benchmark)
    public static class Parameters {
        float[] randomVector;
        MemorySegment onHeapSegment;
        MemorySegment offHeapSegment;

        public Parameters() {
            var randomFloats = new float[16];
            for (int i = 0; i < randomFloats.length; i++) {
                randomFloats[i] = (float) Math.random();
            }
            this.randomVector = randomFloats;
            this.onHeapSegment = MemorySegment.ofArray(randomFloats);
            this.offHeapSegment = Arena.ofAuto().allocate(randomFloats.length * Float.BYTES, 64).copyFrom(onHeapSegment);
        }
    }

    @Benchmark
    @BenchmarkMode(Mode.Throughput)
    @Threads(8)
    public void testLoadArray512(Blackhole bh, Parameters p) {
        bh.consume(FloatVector.SPECIES_512.fromArray(p.randomVector, 0));
    }

    @Benchmark
    @BenchmarkMode(Mode.Throughput)
    @Threads(8)
    public void testLoadOnheapSegment512(Blackhole bh, Parameters p) {
        bh.consume(FloatVector.SPECIES_512.fromMemorySegment(p.onHeapSegment, 0, ByteOrder.LITTLE_ENDIAN));
    }

    @Benchmark
    @BenchmarkMode(Mode.Throughput)
    @Threads(8)
    public void testLoadOffheapSegment512(Blackhole bh, Parameters p) {
        bh.consume(FloatVector.SPECIES_512.fromMemorySegment(p.offHeapSegment, 0, ByteOrder.LITTLE_ENDIAN));
    }

    @Benchmark
    @BenchmarkMode(Mode.Throughput)
    @Threads(8)
    public void testLoadArray256(Blackhole bh, Parameters p) {
        bh.consume(FloatVector.SPECIES_256.fromArray(p.randomVector, 0));
    }

    @Benchmark
    @BenchmarkMode(Mode.Throughput)
    @Threads(8)
    public void testLoadOnheapSegment256(Blackhole bh, Parameters p) {
        bh.consume(FloatVector.SPECIES_256.fromMemorySegment(p.onHeapSegment, 0, ByteOrder.LITTLE_ENDIAN));
    }

    @Benchmark
    @BenchmarkMode(Mode.Throughput)
    @Threads(8)
    public void testLoadOffheapSegment256(Blackhole bh, Parameters p) {
        bh.consume(FloatVector.SPECIES_256.fromMemorySegment(p.offHeapSegment, 0, ByteOrder.LITTLE_ENDIAN));
    }

    @Benchmark
    @BenchmarkMode(Mode.Throughput)
    @Threads(8)
    public void testLoadArray128(Blackhole bh, Parameters p) {
        bh.consume(FloatVector.SPECIES_128.fromArray(p.randomVector, 0));
    }

    @Benchmark
    @BenchmarkMode(Mode.Throughput)
    @Threads(8)
    public void testLoadOnheapSegment128(Blackhole bh, Parameters p) {
        bh.consume(FloatVector.SPECIES_128.fromMemorySegment(p.onHeapSegment, 0, ByteOrder.LITTLE_ENDIAN));
    }

    @Benchmark
    @BenchmarkMode(Mode.Throughput)
    @Threads(8)
    public void testLoadOffheapSegment128(Blackhole bh, Parameters p) {
        bh.consume(FloatVector.SPECIES_128.fromMemorySegment(p.offHeapSegment, 0, ByteOrder.LITTLE_ENDIAN));
    }

    public static void main(String[] args) throws Exception {
        org.openjdk.jmh.Main.main(args);
    }
}


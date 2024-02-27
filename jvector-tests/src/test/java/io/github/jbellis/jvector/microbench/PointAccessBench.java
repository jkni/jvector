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
import java.lang.foreign.ValueLayout;

@Warmup(iterations = 3, time = 5)
@Measurement(iterations = 20, time = 2)
@Fork(warmups = 1, value = 2, jvmArgsAppend = {"--add-modules=jdk.incubator.vector", "--enable-preview", "-Djvector.experimental.enable_native_vectorization=true"})
public class PointAccessBench {
    @State(Scope.Benchmark)
    public static class Parameters {
        float[] randomVector;
        MemorySegment onHeapSegment;
        MemorySegment offHeapSegment;
        byte[] randomIndexes;

        public Parameters() {
            var randomFloats = new float[256 * 128];
            for (int i = 0; i < randomFloats.length; i++) {
                randomFloats[i] = (float) Math.random();
            }
            this.randomVector = randomFloats;
            this.onHeapSegment = MemorySegment.ofArray(randomFloats);
            this.offHeapSegment = Arena.global().allocate(randomFloats.length * Float.BYTES, 64).copyFrom(onHeapSegment).asReadOnly();
            var randomBytes = new byte[128];
            for (int i = 0; i < randomBytes.length; i++) {
                randomBytes[i] = (byte) (Math.random() * randomFloats.length);
            }
            this.randomIndexes = randomBytes;
        }
    }

    @Benchmark
    @BenchmarkMode(Mode.Throughput)
    @Threads(1)
    public void testIndexArray(Blackhole bh, Parameters p) {
        bh.consume(p.randomVector[0 * 256 + Byte.toUnsignedInt(p.randomIndexes[0])]);
    }

    @Benchmark
    @BenchmarkMode(Mode.Throughput)
    @Threads(1)
    public void testIndexOnheapSegment(Blackhole bh, Parameters p) {
        bh.consume(p.onHeapSegment.getAtIndex(ValueLayout.JAVA_FLOAT, 0 * 256 + Byte.toUnsignedInt(p.randomIndexes[0])));
    }

    @Benchmark
    @BenchmarkMode(Mode.Throughput)
    @Threads(1)
    public void testIndexOffheapSegment(Blackhole bh, Parameters p) {
        bh.consume(p.offHeapSegment.getAtIndex(ValueLayout.JAVA_FLOAT, 0 * 256 + Byte.toUnsignedInt(p.randomIndexes[0])));
    }

    public static void main(String[] args) throws Exception {
        org.openjdk.jmh.Main.main(args);
    }
}


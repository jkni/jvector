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
/*

import io.github.jbellis.jvector.vector.DefaultVectorizationProvider;
import io.github.jbellis.jvector.vector.PanamaVectorizationProvider;
import io.github.jbellis.jvector.vector.VectorUtil;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;

@Warmup(iterations = 2, time = 5)
@Measurement(iterations = 3, time = 10)
@Fork(warmups = 1, value = 1, jvmArgsAppend = {"--add-modules=jdk.incubator.vector"})
public class SimdBench {
/*
    private static final DefaultVectorizationProvider java = new DefaultVectorizationProvider();

    static final float[] partials = new float[2048];
    static final int[] shuffle = new int[32 * 128];
    static final float[] results = new float[32];

    static {
        for (int i = 0; i < 32 * 128; i++) {
            shuffle[i] = ThreadLocalRandom.current().nextInt(16);
        }

        for (int i = 0; i < partials.length; i++) {
            partials[i] = ThreadLocalRandom.current().nextFloat();
        }
    }

    @State(Scope.Benchmark)
    public static class Parameters {

    }

    @Benchmark
    @BenchmarkMode(Mode.Throughput)
    @Threads(8)
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    public void shuffleSimd(Blackhole bh, Parameters p) {
        VectorUtil.bulkShuffleSimilarity(shuffle, 128, partials, results);
        bh.consume(results);
    }

    @Benchmark
    @BenchmarkMode(Mode.Throughput)
    @Threads(8)
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    public void shuffleJava(Blackhole bh, Parameters p) {
        java.getVectorUtilSupport().bulkShuffleSimilarity(shuffle, 128, partials, results);
        bh.consume(results);
    }


    public static void main(String[] args) throws Exception {
        org.openjdk.jmh.Main.main(args);
    }
}*/

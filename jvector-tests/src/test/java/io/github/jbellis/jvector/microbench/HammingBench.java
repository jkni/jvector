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


import io.github.jbellis.jvector.vector.DefaultVectorizationProvider;
import io.github.jbellis.jvector.vector.VectorUtil;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;

@Warmup(iterations = 2, time = 5)
@Measurement(iterations = 3, time = 10)
@Fork(warmups = 1, value = 1, jvmArgsAppend = {"--add-modules=jdk.incubator.vector"})
public class HammingBench {

    private static final DefaultVectorizationProvider java = new DefaultVectorizationProvider();

    static int SIZE = 64;
    static final byte[] q1 = new byte[SIZE];
    static final byte[] q2 = new byte[SIZE];


    static {
        ThreadLocalRandom.current().nextBytes(q1);
        ThreadLocalRandom.current().nextBytes(q2);
    }

    @State(Scope.Benchmark)
    public static class Parameters {

    }

    @Benchmark
    @BenchmarkMode(Mode.Throughput)
    @Threads(8)
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    public void zipAndSumSimd(Blackhole bh, Parameters p) {
        bh.consume(VectorUtil.hammingDistance(q1, q2));
    }

    @Benchmark
    @BenchmarkMode(Mode.Throughput)
    @Threads(8)
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    public void zzipAndSumJava(Blackhole bh, Parameters p) {
        bh.consume(java.getVectorUtilSupport().hammingDistance(q1, q2));
    }

   /* @Benchmark
    @BenchmarkMode(Mode.Throughput)
    @Threads(8)
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    public void dotProduct(Blackhole bh, Parameters p) {
        bh.consume(VectorUtil.dotProduct(q3, 0, q1, 22, q3.length));
    }*/

    public static void main(String[] args) throws Exception {
        org.openjdk.jmh.Main.main(args);
    }
}

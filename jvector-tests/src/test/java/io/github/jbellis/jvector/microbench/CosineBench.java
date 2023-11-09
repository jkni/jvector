package io.github.jbellis.jvector.microbench;

import io.github.jbellis.jvector.vector.DefaultVectorizationProvider;
import io.github.jbellis.jvector.vector.VectorUtil;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
@Warmup(iterations = 2, time = 5)
@Measurement(iterations = 3, time = 10)
@Fork(warmups = 1, value = 1, jvmArgsAppend = {"--add-modules=jdk.incubator.vector"})
public class CosineBench {
    static double[] cachedCosine = new double[65];
    static final Random r = new Random();



    static {
        for (int i = 0; i < 65; i++) {
            cachedCosine[i] = Math.cos(Math.PI * i / 64);
        }
    }

    @State(Scope.Benchmark)
    public static class Parameters {
        int i = 0;
    }

    @Benchmark
    @BenchmarkMode(Mode.Throughput)
    @Threads(8)
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    public void cachedCosine(Blackhole bh, CosineBench.Parameters p) {
        p.i = (p.i + 1) % 65;
        bh.consume(cachedCosine[p.i]);
    }

    @Benchmark
    @BenchmarkMode(Mode.Throughput)
    @Threads(8)
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    public void cosine(Blackhole bh, CosineBench.Parameters p) {
        p.i = (p.i + 1) % 65;
        bh.consume(Math.cos(Math.PI * p.i / 64));
    }

    public static void main(String[] args) throws Exception {
        org.openjdk.jmh.Main.main(args);
    }
}

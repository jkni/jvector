package io.github.jbellis.jvector.disk;

import com.carrotsearch.randomizedtesting.RandomizedTest;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.pq.OnDiskADCGraphIndex;
import io.github.jbellis.jvector.pq.PQVectors;
import io.github.jbellis.jvector.pq.ProductQuantization;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static io.github.jbellis.jvector.TestUtil.createRandomVectors;
import static org.junit.Assert.assertEquals;

@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestOnDiskADCGraphIndex extends RandomizedTest {

    private Path testDirectory;

    private TestUtil.FullyConnectedGraphIndex<VectorFloat<?>> fullyConnectedGraph;
    private TestUtil.RandomlyConnectedGraphIndex<VectorFloat<?>> randomlyConnectedGraph;

    @Before
    public void setup() throws IOException {
        testDirectory = Files.createTempDirectory(this.getClass().getSimpleName());
    }

    @After
    public void tearDown() {
        TestUtil.deleteQuietly(testDirectory);
    }

    @Test
    public void testFusedGraph() throws Exception {
        // generate random graph, M=32, 256-dimension vectors
        var graph = new TestUtil.RandomlyConnectedGraphIndex<VectorFloat<?>>(100_000, 32, getRandom());
        var outputPath = testDirectory.resolve("large_graph");
        var vectors = createRandomVectors(100_000, 256);
        var ravv = new ListRandomAccessVectorValues(vectors, 256);
        var pq = ProductQuantization.compute(ravv, 64, 32, false);
        var compressed = pq.encodeAll(vectors);
        var pqv = new PQVectors(pq, compressed);

        TestUtil.writeFusedGraph(graph, ravv, pqv, outputPath);

        try (var marr = new SimpleMappedReader(outputPath.toAbsolutePath().toString());
             var onDiskGraph = new OnDiskADCGraphIndex<VectorFloat<?>>(marr::duplicate, 0);
             var cachedOnDiskGraph = new CachingFusedGraphIndex(onDiskGraph))
        {
            TestUtil.assertGraphEquals(graph, onDiskGraph);
            TestUtil.assertGraphEquals(graph, cachedOnDiskGraph);
            try (var cachedOnDiskView = cachedOnDiskGraph.getView())
            {
                var queryVector = TestUtil.randomVector(getRandom(), 256);
                var fusedScoreFunction = cachedOnDiskGraph.approximateFusedScoreFunctionFor(pqv, queryVector, VectorSimilarityFunction.DOT_PRODUCT);
                var scoreFunction = pqv.approximateScoreFunctionFor(queryVector, VectorSimilarityFunction.DOT_PRODUCT);
                for (int i = 0; i < 100_000; i++) {
                    var bulkSimilarities = fusedScoreFunction.bulkSimilarityTo(i);
                    var neighbors = cachedOnDiskView.getNeighborsIterator(i);
                    for (int j = 0; neighbors.hasNext(); j++) {
                        var neighbor = neighbors.next();
                        assertEquals(scoreFunction.similarityTo(neighbor), bulkSimilarities.get(j), 0.0001);
                    }
                }

            }
        }


    }
}

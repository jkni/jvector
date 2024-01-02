/*
 * All changes to the original code are Copyright DataStax, Inc.
 *
 * Please see the included license file for details.
 */

/*
 * Original license:
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.github.jbellis.jvector.graph;

import com.carrotsearch.randomizedtesting.RandomizedTest;
import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.util.FixedBitSet;
import io.github.jbellis.jvector.vector.VectorEncoding;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.junit.Before;

import java.util.Arrays;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Tests KNN graphs
 */
public class TestFloatVectorGraph extends GraphIndexTestCase<VectorFloat<?>> {
    protected static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();
    @Before
    public void setup() {
        similarityFunction = RandomizedTest.randomFrom(VectorSimilarityFunction.values());
    }

    @Override
    VectorEncoding getVectorEncoding() {
        return VectorEncoding.FLOAT32;
    }

    @Override
    VectorFloat<?> randomVector(int dim) {
        return TestUtil.randomVector(getRandom(), dim);
    }

    @Override
    AbstractMockVectorValues<VectorFloat<?>> vectorValues(int size, int dimension) {
        return MockVectorValues.fromValues(createRandomFloatVectors(size, dimension, getRandom()));
    }

    @Override
    AbstractMockVectorValues<VectorFloat<?>> vectorValues(VectorFloat<?>[] values) {
        return MockVectorValues.fromValues(values);
    }

    @Override
    RandomAccessVectorValues<VectorFloat<?>> circularVectorValues(int nDoc) {
        return new CircularFloatVectorValues(nDoc);
    }

    @Override
    VectorFloat<?> getTargetVector() {
        return vectorTypeSupport.createFloatType(new float[] {1f, 0f});
    }

    public void testSearchWithSkewedAcceptOrds() {
        int nDoc = 1000;
        similarityFunction = VectorSimilarityFunction.EUCLIDEAN;
        RandomAccessVectorValues<VectorFloat<?>> vectors = circularVectorValues(nDoc);
        VectorEncoding vectorEncoding = getVectorEncoding();
        getRandom().nextInt();
        GraphIndexBuilder<VectorFloat<?>> builder = new GraphIndexBuilder<>(vectors, vectorEncoding, similarityFunction, 16, 100, 1.0f, 1.0f);
        var graph = TestUtil.buildSequentially(builder, vectors);

        // Skip over half of the documents that are closest to the query vector
        FixedBitSet acceptOrds = new FixedBitSet(nDoc);
        for (int i = 500; i < nDoc; i++) {
            acceptOrds.set(i);
        }
        SearchResult.NodeScore[] nn =
                GraphSearcher.search(
                        getTargetVector(),
                        10,
                        vectors.copy(),
                        getVectorEncoding(),
                        similarityFunction,
                        graph,
                        acceptOrds
                ).getNodes();

        int[] nodes = Arrays.stream(nn).mapToInt(nodeScore -> nodeScore.node).toArray();
        assertEquals("Number of found results is not equal to [10].", 10, nodes.length);
        int sum = 0;
        for (int node : nodes) {
            assertTrue("the results include a deleted document: " + node, acceptOrds.get(node));
            sum += node;
        }
        // We still expect to get reasonable recall. The lowest non-skipped docIds
        // are closest to the query vector: sum(500,509) = 5045
        assertTrue("sum(result docs)=" + sum, sum < 5100);
    }
}

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

import com.carrotsearch.randomizedtesting.RandomizedTest;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import org.junit.Test;

import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.junit.Assert.assertEquals;

@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestProductQuantization extends RandomizedTest {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();
    @Test
    public void testPerfectReconstruction() {
        var vectors = IntStream.range(0, ProductQuantization.DEFAULT_CLUSTERS).mapToObj(
                i -> vectorTypeSupport.createFloatType(new float[] {getRandom().nextInt(100000), getRandom().nextInt(100000), getRandom().nextInt(100000) }))
                .collect(Collectors.<VectorFloat<?>>toList());
        var ravv = new ListRandomAccessVectorValues(vectors, 3);
        var pq = ProductQuantization.compute(ravv, 2, false);
        var encoded = pq.encodeAll(vectors);
        var decodedScratch = vectorTypeSupport.createFloatType(3);
        // if the number of vectors is equal to the number of clusters, we should perfectly reconstruct vectors
        for (int i = 0; i < vectors.size(); i++) {
            pq.decode(encoded[i], decodedScratch);
            assertEquals(vectors.get(i), decodedScratch);
        }
    }
}

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

import io.github.jbellis.jvector.util.ArrayUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

public class MockVectorValues extends AbstractMockVectorValues<VectorFloat<?>> {
    private final VectorFloat<?> scratch;

    public static MockVectorValues fromValues(VectorFloat<?>[] values) {
        return new MockVectorValues(values[0].length(), values);
    }

    MockVectorValues(int dimension, VectorFloat<?>[] denseValues) {
        super(dimension, denseValues);
        this.scratch = vectorTypeSupport.createFloatType(dimension);
    }

    @Override
    public MockVectorValues copy() {
        return new MockVectorValues(dimension,
                                    ArrayUtil.copyOfSubArray(denseValues, 0, denseValues.length)
        );
    }

    @Override
    public boolean isValueShared() {
        return true;
    }

    @Override
    public VectorFloat<?> vectorValue(int targetOrd) {
        VectorFloat<?> original = super.vectorValue(targetOrd);
        // present a single vector reference to callers like the disk-backed RAVV implmentations,
        // to catch cases where they are not making a copy
           for (int i = 0; i < dimension; i++) {
                scratch.set(i, original.get(i));
            }
        return scratch;
    }
}

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

import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

abstract class AbstractMockVectorValues<T> implements RandomAccessVectorValues<T> {
    protected static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();
    protected final int dimension;
    protected final T[] denseValues;

    AbstractMockVectorValues(int dimension, T[] denseValues) {
        for (var a : denseValues) {
            assert a != null;
        }
        this.dimension = dimension;
        this.denseValues = denseValues;
    }

    @Override
    public int size() {
        return denseValues.length;
    }

    @Override
    public int dimension() {
        return dimension;
    }

    @Override
    public boolean isValueShared() {
        return false;
    }

    @Override
    public T vectorValue(int targetOrd) {
        return denseValues[targetOrd];
    }

    @Override
    public abstract AbstractMockVectorValues<T> copy();
}

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

package com.github.jbellis.jvector.graph;

import com.github.jbellis.jvector.annotations.Unshared;
import com.github.jbellis.jvector.util.BytesRef;

import static com.github.jbellis.jvector.util.DocIdSetIterator.NO_MORE_DOCS;

abstract class AbstractMockVectorValues<T> implements RandomAccessVectorValues<T> {

  protected final int dimension;
  protected final T[] denseValues;
  protected final T[] values;
  protected final int numVectors;
  protected final BytesRef binaryValue;

  private long callingThreadID = -1;
  protected int pos = -1;

  AbstractMockVectorValues(T[] values, int dimension, T[] denseValues, int numVectors) {
    this.dimension = dimension;
    this.values = values;
    this.denseValues = denseValues;
    // used by tests that build a graph from bytes rather than floats
    binaryValue = new BytesRef(dimension);
    binaryValue.length = dimension;
    this.numVectors = numVectors;
  }

  @Override
  public int size() {
    return numVectors;
  }

  @Override
  public int dimension() {
    return dimension;
  }

  @Override
  @Unshared
  public T vectorValue(int targetOrd) {
    if (callingThreadID < 0) {
      callingThreadID = Thread.currentThread().getId();
    }
    if (callingThreadID != Thread.currentThread().getId()) {
      throw new RuntimeException(
          "RandomAccessVectorValues is not thread safe, but multiple calling threads detected");
    }

    return denseValues[targetOrd];
  }

  @Override
  public abstract AbstractMockVectorValues<T> copy();

  public abstract T vectorValue();

  private boolean seek(int target) {
    if (target >= 0 && target < values.length && values[target] != null) {
      pos = target;
      return true;
    } else {
      return false;
    }
  }

  public int docID() {
    return pos;
  }

  public int nextDoc() {
    return advance(pos + 1);
  }

  public int advance(int target) {
    while (++pos < values.length) {
      if (seek(pos)) {
        return pos;
      }
    }
    return NO_MORE_DOCS;
  }
}

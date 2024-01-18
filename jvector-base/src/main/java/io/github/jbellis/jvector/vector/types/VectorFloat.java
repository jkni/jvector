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

package io.github.jbellis.jvector.vector.types;

public interface VectorFloat<T> extends VectorType<Float, T>
{
    @Override
    VectorFloat<T> copy();

    void copyFrom(VectorFloat<?> src, int srcOffset, int destOffset, int length);

    float get(int i);

    void set(int i, float value);

    void zero();
}

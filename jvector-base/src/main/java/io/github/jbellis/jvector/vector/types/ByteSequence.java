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

import io.github.jbellis.jvector.util.Accountable;

public interface ByteSequence<T> extends Accountable
{
    /**
     * @return entire sequence backing storage
     */
    T get();

    int length();

    byte get(int i);

    void set(int i, byte value);

    /**
     * @param byteIndex byte index inside the sequence to start setting short value
     * @param value short value to set
     */
    void setLittleEndianShort(int byteIndex, short value);

    void zero();

    void copyFrom(ByteSequence<?> src, int srcOffset, int destOffset, int length);

    ByteSequence<T> copy();
}

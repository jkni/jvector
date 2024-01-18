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

package io.github.jbellis.jvector.vector;

import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.vector.types.VectorByte;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.DataOutput;
import java.io.IOException;
import java.nio.Buffer;

/**
 * VectorTypeSupport using off-heap MemorySegments.
 */
public class OffHeapVectorProvider implements VectorTypeSupport
{
    @Override
    public VectorFloat<?> createFloatType(Object data)
    {
        if (data instanceof Buffer)
            return new OffHeapVectorFloat((Buffer) data);

        return new OffHeapVectorFloat((float[]) data);
    }

    @Override
    public VectorFloat<?> createFloatType(int length)
    {
        return new OffHeapVectorFloat(length);
    }

    @Override
    public VectorFloat<?> readFloatType(RandomAccessReader r, int size) throws IOException
    {
        float[] d = new float[size];
        r.readFully(d);
        return new OffHeapVectorFloat(d);
    }

    @Override
    public void writeFloatType(DataOutput out, VectorFloat<?> vector) throws IOException
    {
        for (int i = 0; i < vector.length(); i++)
            out.writeFloat(vector.get(i));
    }

    @Override
    public VectorByte<?> createByteType(Object data)
    {
        if (data instanceof Buffer)
            return new OffHeapVectorByte((Buffer) data);

        return new OffHeapVectorByte((byte[]) data);
    }

    @Override
    public VectorByte<?> createByteType(int length)
    {
        return new OffHeapVectorByte(length);
    }

    @Override
    public VectorByte<?> readByteType(RandomAccessReader r, int size) throws IOException
    {
        var vector = new OffHeapVectorByte(size);
        r.readFully(vector.get().asByteBuffer());
        return vector;
    }

    @Override
    public void readByteType(RandomAccessReader r, VectorByte<?> vector) throws IOException {
        r.readFully(((OffHeapVectorByte) vector).get().asByteBuffer());
    }


    @Override
    public void writeByteType(DataOutput out, VectorByte<?> vector) throws IOException
    {
        for (int i = 0; i < vector.length(); i++)
            out.writeByte(vector.get(i));
    }
}

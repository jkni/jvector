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

import io.github.jbellis.jvector.vector.types.VectorByte;
import io.github.jbellis.jvector.vector.types.VectorType;

import java.lang.foreign.Arena;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SegmentAllocator;
import java.lang.foreign.ValueLayout;
import java.nio.Buffer;

/**
 * VectorByte implementation backed by an off-heap MemorySegment.
 */
public class OffHeapVectorByte implements VectorByte<MemorySegment> {
    private final MemorySegment segment;
    private static final ThreadLocal<SegmentAllocator> allocator =
            ThreadLocal.withInitial(() -> SegmentAllocator.slicingAllocator(Arena.ofAuto().allocate(1024 * 1024 * 128L, 64))); // TODO: real buffer pool
    private final int length;

    OffHeapVectorByte(int length) {
        MemorySegment segment;
        try {
            segment = allocator.get().allocate(length, 64);
        } catch (IndexOutOfBoundsException e) {
            allocator.set(SegmentAllocator.slicingAllocator(Arena.ofAuto().allocate(1024 * 1024 * 128L, 64)));
            segment = allocator.get().allocate(length, 64);
        }
        this.segment = segment;
        this.length = length;
    }

    OffHeapVectorByte(Buffer data) {
        this(data.remaining());
        segment.copyFrom(MemorySegment.ofBuffer(data));
    }

    OffHeapVectorByte(byte[] data) {
        this(data.length);
        segment.copyFrom(MemorySegment.ofArray(data));
    }

    @Override
    public long ramBytesUsed() {
        return MemoryLayout.sequenceLayout(length, ValueLayout.JAVA_BYTE).byteSize();
    }

    @Override
    public void copyFrom(VectorByte<?> src, int srcOffset, int destOffset, int length) {
        OffHeapVectorByte csrc = (OffHeapVectorByte) src;
        segment.asSlice(destOffset, length).copyFrom(csrc.segment.asSlice(srcOffset));
    }

    @Override
    public VectorEncoding type() {
        return VectorEncoding.BYTE;
    }

    @Override
    public MemorySegment get() {
        return segment;
    }

    @Override
    public byte get(int n) {
        return segment.get(ValueLayout.JAVA_BYTE, n);
    }

    @Override
    public void set(int n, byte value) {
        segment.set(ValueLayout.JAVA_BYTE, n, value);
    }

    @Override
    public void zero() {
        segment.fill((byte) 0);
    }

    @Override
    public int length() {
        return (int) segment.byteSize();
    }

    @Override
    public VectorType<Byte, MemorySegment> copy() {
        OffHeapVectorByte copy = new OffHeapVectorByte(length());
        copy.copyFrom(this, 0, 0, length());
        return copy;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        OffHeapVectorByte that = (OffHeapVectorByte) o;
        return segment.mismatch(that.segment) == -1;
    }

    @Override
    public int hashCode() {
        return segment.hashCode();
    }
}

package io.github.jbellis.jvector.vector;

import java.lang.foreign.Arena;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SegmentAllocator;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Objects;

import io.github.jbellis.jvector.vector.types.VectorFloat;

final public class OffHeapVectorFloat implements VectorFloat<MemorySegment>
{
    private final MemorySegment segment;
    private static final ThreadLocal<SegmentAllocator> allocator =
            ThreadLocal.withInitial(() -> SegmentAllocator.slicingAllocator(Arena.ofAuto().allocate(1024 * 1024 * 128L, 64)));
    private final int length;

    OffHeapVectorFloat(int length) {
        MemorySegment segment;
        try {
            segment = allocator.get().allocate(length * Float.BYTES, 64);
        } catch (IndexOutOfBoundsException e) {
            allocator.set(SegmentAllocator.slicingAllocator(Arena.ofAuto().allocate(1024 * 1024 * 128L, 64)));
            segment = allocator.get().allocate(length * Float.BYTES, 64);
        }
        this.segment = segment;
        this.length = length;
    }

    OffHeapVectorFloat(ByteBuffer buffer) {
        this(buffer.remaining());
        segment.copyFrom(MemorySegment.ofBuffer(buffer));
    }

    OffHeapVectorFloat(float[] data) {
        this(data.length);
        segment.copyFrom(MemorySegment.ofArray(data));
    }

    @Override
    public long ramBytesUsed()
    {
        return MemoryLayout.sequenceLayout(length, ValueLayout.JAVA_FLOAT).byteSize();
    }

    @Override
    public VectorEncoding type()
    {
        return VectorEncoding.FLOAT32;
    }

    @Override
    public MemorySegment get()
    {
        return segment;
    }

    @Override
    public float get(int n)
    {
        return segment.getAtIndex(ValueLayout.JAVA_FLOAT, n);
    }

    @Override
    public void set(int n, float value)
    {
        segment.setAtIndex(ValueLayout.JAVA_FLOAT, n, value);
    }

    @Override
    public int length() {
        return length;
    }

    @Override
    public int offset(int i)
    {
        return i * Float.BYTES;
    }

    @Override
    public VectorFloat<MemorySegment> copy()
    {
        OffHeapVectorFloat copy = new OffHeapVectorFloat(length());
        copy.copyFrom(this, 0, 0, length());
        return copy;
    }

    @Override
    public void copyFrom(VectorFloat<?> src, int srcOffset, int destOffset, int length)
    {
        OffHeapVectorFloat csrc = (OffHeapVectorFloat) src;
        segment.asSlice((long) destOffset * Float.BYTES, (long) length * Float.BYTES)
                .copyFrom(csrc.segment.asSlice((long) srcOffset * Float.BYTES, (long) length * Float.BYTES));
    }

    @Override
    public float[] array()
    {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean equals(Object o)
    {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        OffHeapVectorFloat that = (OffHeapVectorFloat) o;
        return segment.equals(that.segment);
    }

    @Override
    public int hashCode() {
        return segment.hashCode();
    }
}

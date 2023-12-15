package io.github.jbellis.jvector.vector;

import java.lang.foreign.Arena;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SegmentAllocator;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Objects;

import io.github.jbellis.jvector.vector.types.VectorByte;
import io.github.jbellis.jvector.vector.types.VectorType;

public class OffHeapVectorByte implements VectorByte<MemorySegment> {
    private final MemorySegment segment;
    private static final ThreadLocal<SegmentAllocator> allocator =
            ThreadLocal.withInitial(() -> SegmentAllocator.slicingAllocator(Arena.ofAuto().allocate(1024 * 1024 * 128L, 64)));
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

    OffHeapVectorByte(ByteBuffer data) {
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
    public byte[] array() {
        throw new UnsupportedOperationException();
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
        return segment.equals(that.segment);
    }

    @Override
    public int hashCode() {
        return segment.hashCode();
    }
}

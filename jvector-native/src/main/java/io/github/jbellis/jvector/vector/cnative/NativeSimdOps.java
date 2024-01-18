// Generated by jextract

package io.github.jbellis.jvector.vector.cnative;

import java.lang.foreign.AddressLayout;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;

import static java.lang.foreign.ValueLayout.*;

public class NativeSimdOps  {

    public static final OfByte C_CHAR = JAVA_BYTE;
    public static final OfShort C_SHORT = JAVA_SHORT;
    public static final OfInt C_INT = JAVA_INT;
    public static final OfLong C_LONG = JAVA_LONG;
    public static final OfLong C_LONG_LONG = JAVA_LONG;
    public static final OfFloat C_FLOAT = JAVA_FLOAT;
    public static final OfDouble C_DOUBLE = JAVA_DOUBLE;
    public static final AddressLayout C_POINTER = RuntimeHelper.POINTER;
    public static MethodHandle dot_product_f32$MH() {
        return RuntimeHelper.requireNonNull(constants$0.const$1,"dot_product_f32");
    }
    /**
     * {@snippet :
     * float dot_product_f32(int preferred_size, float* a, int aoffset, float* b, int boffset, int length);
     * }
     */
    public static float dot_product_f32(int preferred_size, MemorySegment a, int aoffset, MemorySegment b, int boffset, int length) {
        var mh$ = dot_product_f32$MH();
        try {
            return (float)mh$.invokeExact(preferred_size, a, aoffset, b, boffset, length);
        } catch (Throwable ex$) {
            throw new AssertionError("should not reach here", ex$);
        }
    }
    public static MethodHandle bulk_shuffle_dot_f32_512$MH() {
        return RuntimeHelper.requireNonNull(constants$0.const$3,"bulk_shuffle_dot_f32_512");
    }
    /**
     * {@snippet :
     * void bulk_shuffle_dot_f32_512(unsigned char* shuffles, int codebookCount, float* partials, float* results);
     * }
     */
    public static void bulk_shuffle_dot_f32_512(MemorySegment shuffles, int codebookCount, MemorySegment partials, MemorySegment results) {
        var mh$ = bulk_shuffle_dot_f32_512$MH();
        try {
            mh$.invokeExact(shuffles, codebookCount, partials, results);
        } catch (Throwable ex$) {
            throw new AssertionError("should not reach here", ex$);
        }
    }
    public static MethodHandle bulk_shuffle_euclidean_f32_512$MH() {
        return RuntimeHelper.requireNonNull(constants$0.const$4,"bulk_shuffle_euclidean_f32_512");
    }
    /**
     * {@snippet :
     * void bulk_shuffle_euclidean_f32_512(unsigned char* shuffles, int codebookCount, float* partials, float* results);
     * }
     */
    public static void bulk_shuffle_euclidean_f32_512(MemorySegment shuffles, int codebookCount, MemorySegment partials, MemorySegment results) {
        var mh$ = bulk_shuffle_euclidean_f32_512$MH();
        try {
            mh$.invokeExact(shuffles, codebookCount, partials, results);
        } catch (Throwable ex$) {
            throw new AssertionError("should not reach here", ex$);
        }
    }
}



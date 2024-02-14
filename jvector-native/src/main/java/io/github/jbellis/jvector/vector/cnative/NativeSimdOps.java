// Generated by jextract

package io.github.jbellis.jvector.vector.cnative;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.VarHandle;
import java.nio.ByteOrder;
import java.lang.foreign.*;
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
    /**
     * {@snippet :
     * #define true 1
     * }
     */
    public static int true_() {
        return (int)1L;
    }
    /**
     * {@snippet :
     * #define false 0
     * }
     */
    public static int false_() {
        return (int)0L;
    }
    /**
     * {@snippet :
     * #define __bool_true_false_are_defined 1
     * }
     */
    public static int __bool_true_false_are_defined() {
        return (int)1L;
    }
    public static MethodHandle check_compatibility$MH() {
        return RuntimeHelper.requireNonNull(constants$0.const$1,"check_compatibility");
    }
    /**
     * {@snippet :
     * _Bool check_compatibility(,...);
     * }
     */
    public static boolean check_compatibility(Object... x0) {
        var mh$ = check_compatibility$MH();
        try {
            return (boolean)mh$.invokeExact(x0);
        } catch (Throwable ex$) {
            throw new AssertionError("should not reach here", ex$);
        }
    }
    public static MethodHandle dot_product_f32$MH() {
        return RuntimeHelper.requireNonNull(constants$0.const$3,"dot_product_f32");
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
    public static MethodHandle euclidean_f32$MH() {
        return RuntimeHelper.requireNonNull(constants$0.const$4,"euclidean_f32");
    }
    /**
     * {@snippet :
     * float euclidean_f32(int preferred_size, float* a, int aoffset, float* b, int boffset, int length);
     * }
     */
    public static float euclidean_f32(int preferred_size, MemorySegment a, int aoffset, MemorySegment b, int boffset, int length) {
        var mh$ = euclidean_f32$MH();
        try {
            return (float)mh$.invokeExact(preferred_size, a, aoffset, b, boffset, length);
        } catch (Throwable ex$) {
            throw new AssertionError("should not reach here", ex$);
        }
    }
    public static MethodHandle bulk_shuffle_dot_f32_512$MH() {
        return RuntimeHelper.requireNonNull(constants$0.const$6,"bulk_shuffle_dot_f32_512");
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
        return RuntimeHelper.requireNonNull(constants$1.const$0,"bulk_shuffle_euclidean_f32_512");
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
    public static MethodHandle assemble_and_sum_f32_512$MH() {
        return RuntimeHelper.requireNonNull(constants$1.const$2,"assemble_and_sum_f32_512");
    }
    /**
     * {@snippet :
     * float assemble_and_sum_f32_512(float* data, int dataBase, unsigned char* baseOffsets, int baseOffsetsLength);
     * }
     */
    public static float assemble_and_sum_f32_512(MemorySegment data, int dataBase, MemorySegment baseOffsets, int baseOffsetsLength) {
        var mh$ = assemble_and_sum_f32_512$MH();
        try {
            return (float)mh$.invokeExact(data, dataBase, baseOffsets, baseOffsetsLength);
        } catch (Throwable ex$) {
            throw new AssertionError("should not reach here", ex$);
        }
    }
    public static MethodHandle calculate_partial_sums_dot_f32_512$MH() {
        return RuntimeHelper.requireNonNull(constants$1.const$4,"calculate_partial_sums_dot_f32_512");
    }
    /**
     * {@snippet :
     * void calculate_partial_sums_dot_f32_512(float* codebook, int codebookBase, int size, int clusterCount, float* query, int queryOffset, float* partialSums);
     * }
     */
    public static void calculate_partial_sums_dot_f32_512(MemorySegment codebook, int codebookBase, int size, int clusterCount, MemorySegment query, int queryOffset, MemorySegment partialSums) {
        var mh$ = calculate_partial_sums_dot_f32_512$MH();
        try {
            mh$.invokeExact(codebook, codebookBase, size, clusterCount, query, queryOffset, partialSums);
        } catch (Throwable ex$) {
            throw new AssertionError("should not reach here", ex$);
        }
    }
    public static MethodHandle calculate_partial_sums_euclidean_f32_512$MH() {
        return RuntimeHelper.requireNonNull(constants$1.const$5,"calculate_partial_sums_euclidean_f32_512");
    }
    /**
     * {@snippet :
     * void calculate_partial_sums_euclidean_f32_512(float* codebook, int codebookBase, int size, int clusterCount, float* query, int queryOffset, float* partialSums);
     * }
     */
    public static void calculate_partial_sums_euclidean_f32_512(MemorySegment codebook, int codebookBase, int size, int clusterCount, MemorySegment query, int queryOffset, MemorySegment partialSums) {
        var mh$ = calculate_partial_sums_euclidean_f32_512$MH();
        try {
            mh$.invokeExact(codebook, codebookBase, size, clusterCount, query, queryOffset, partialSums);
        } catch (Throwable ex$) {
            throw new AssertionError("should not reach here", ex$);
        }
    }
}



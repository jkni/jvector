#include <stdio.h>
#include <immintrin.h>
#include <inttypes.h>
#include <math.h>
#include "jvector_simd.h"

float dot_product_f32_64(const float* a, int aoffset, const float* b, int boffset) {

     __m128 va = _mm_castsi128_ps(_mm_loadl_epi64((__m128i *)(a + aoffset)));
     __m128 vb = _mm_castsi128_ps(_mm_loadl_epi64((__m128i *)(b + boffset)));
     __m128 r  = _mm_mul_ps(va, vb); // Perform element-wise multiplication

    // Horizontal sum of the vector to get dot product
    __attribute__((aligned(16))) float result[4];
    _mm_store_ps(result, r);
    return result[0] + result[1];
}

float dot_product_f32_256(const float* a, int aoffset, const float* b, int boffset, int length) {
    float dot = 0.0;
    int ao = aoffset;
    int bo = boffset;
    int alim = aoffset + length;
    int blim = boffset + length;

    if (length >= 8) {
        __m256 sum = _mm256_setzero_ps();

        for(; ao < alim && bo < blim; ao += 8, bo += 8) {
            // Load float32
            __m256 va = _mm256_loadu_ps(a + ao);
            __m256 vb = _mm256_loadu_ps(b + bo);

            // Multiply and accumulate
            sum = _mm256_fmadd_ps(va, vb, sum);
        }

        // Horizontal sum of the vector to get dot product
        __attribute__((aligned(16))) float result[8];
        _mm256_store_ps(result, sum);

        for(int i = 0; i < 8; ++i) {
            dot += result[i];
        }
    }

    for (; ao < alim && bo < blim; ao++, bo++) {
        dot += a[ao] * b[bo];
    }

    return dot;
}

float dot_product_f32_512(const float* a, int aoffset, const float* b, int boffset, int length) {
#if defined(__AVX512F__)
    float dot = 0.0;
    int ao = aoffset;
    int bo = boffset;
    int alim = aoffset + length;
    int blim = boffset + length;

    if (length >= 16) {
        __m512 sum = _mm512_setzero_ps();

        for(; ao < alim && bo < blim; ao += 16, bo += 16) {
            // Load float32
            __m512 va = _mm512_loadu_ps(a + ao);
            __m512 vb = _mm512_loadu_ps(b + bo);

            // Multiply and accumulate
            sum = _mm512_fmadd_ps(va, vb, sum);
        }

        // Horizontal sum of the vector to get dot product
        __attribute__((aligned(16)))  float result[16];
        _mm512_store_ps(result, sum);

        for(int i = 0; i < 16; ++i) {
            dot += result[i];
        }
    }

    for (; ao < alim && bo < blim; ao++, bo++) {
        dot += a[ao] * b[bo];
    }

    return dot;
#else
    return dot_product_f32_256(a, aoffset, b, boffset, length);
#endif
}

float dot_product_f32(int preferred_size, const float* a, int aoffset, const float* b, int boffset, int length) {
    if (length == 2)
        return dot_product_f32_64(a, aoffset, b, boffset);

    return (preferred_size == 512 && length >= 16)
           ? dot_product_f32_512(a, aoffset, b, boffset, length)
           : dot_product_f32_256(a, aoffset, b, boffset, length);
}


// We're reimplementing the following Java
/*    public static void bulkShuffleSimilarity(OffHeapVectorByte shuffles, int codebookCount, OffHeapVectorFloat tlPartials, OffHeapVectorFloat results) {
        // 32 is from neighbor count
        // 16 is from CLUSTERS
        var tmpLeft = FloatVector.zero(FloatVector.SPECIES_512);
        var tmpRight = FloatVector.zero(FloatVector.SPECIES_512);
        var intShuffles = new int[shuffles.length()];
        for (int i = 0; i < shuffles.length(); i++) {
            intShuffles[i] = Byte.toUnsignedInt(shuffles.get(i));
        }
        for (int i = 0; i < codebookCount; i++) {
            var shuffleLeft = VectorShuffle.fromArray(FloatVector.SPECIES_512, intShuffles, i * 32);
            var shuffleRight = VectorShuffle.fromArray(FloatVector.SPECIES_512, intShuffles, i * 32 + 16);
            var partials = FloatVector.fromMemorySegment(FloatVector.SPECIES_512, tlPartials.get(), tlPartials.offset(i * 16), ByteOrder.LITTLE_ENDIAN);
            tmpLeft = tmpLeft.add(partials.rearrange(shuffleLeft));
            tmpRight = tmpRight.add(partials.rearrange(shuffleRight));
        }
        tmpLeft = tmpLeft.add(1);
        tmpRight = tmpRight.add(1);
        tmpLeft = tmpLeft.div(2);
        tmpRight = tmpRight.div(2);
        tmpLeft.intoMemorySegment(results.get(), 0, ByteOrder.LITTLE_ENDIAN);
        tmpRight.intoMemorySegment(results.get(), results.offset(16), ByteOrder.LITTLE_ENDIAN);
    }*/
void bulk_shuffle_similarity_f32_512(const unsigned char* shuffles, int codebookCount, const float* partials, float* results) {
    __m128i shuffleLeftRaw;
    __m128i shuffleRightRaw;
    __m512i shuffleLeft;
    __m512i shuffleRight;
    __m512 tmpLeft = _mm512_setzero_ps();
    __m512 tmpRight = _mm512_setzero_ps();

    for (int i = 0; i < codebookCount; i++) {
        shuffleLeftRaw = _mm_loadu_si128((__m128i *)(shuffles + i * 32));
        shuffleRightRaw = _mm_loadu_si128((__m128i *)(shuffles + i * 32 + 16));
        shuffleLeft = _mm512_cvtepu8_epi32(shuffleLeftRaw);
        shuffleRight = _mm512_cvtepu8_epi32(shuffleRightRaw);
        __m512 partialsVec = _mm512_loadu_ps(partials + i * 16);
        tmpLeft = _mm512_add_ps(tmpLeft, _mm512_permutexvar_ps(shuffleLeft, partialsVec));
        tmpRight = _mm512_add_ps(tmpRight, _mm512_permutexvar_ps(shuffleRight, partialsVec));
    }

    tmpLeft = _mm512_add_ps(tmpLeft, _mm512_set1_ps(1.0));
    tmpRight = _mm512_add_ps(tmpRight, _mm512_set1_ps(1.0));
    tmpLeft = _mm512_div_ps(tmpLeft, _mm512_set1_ps(2.0));
    tmpRight = _mm512_div_ps(tmpRight, _mm512_set1_ps(2.0));

    _mm512_storeu_ps(results, tmpLeft);
    _mm512_storeu_ps(results + 16, tmpRight);
}


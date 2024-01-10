#ifndef VECTOR_SIMD_DOT_H
#define VECTOR_SIMD_DOT_H



//F32
float dot_product_f32(int preferred_size, const float* a, int aoffset, const float* b, int boffset, int length);
void bulk_shuffle_dot_f32_512(const unsigned char* shuffles, int codebookCount, const float* partials, float* results);
void bulk_shuffle_euclidean_f32_512(const unsigned char* shuffles, int codebookCount, const float* partials, float* results);
#endif
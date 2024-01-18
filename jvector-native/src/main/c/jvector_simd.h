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

#ifndef VECTOR_SIMD_DOT_H
#define VECTOR_SIMD_DOT_H



//F32
float dot_product_f32(int preferred_size, const float* a, int aoffset, const float* b, int boffset, int length);
void bulk_shuffle_dot_f32_512(const unsigned char* shuffles, int codebookCount, const float* partials, float* results);
void bulk_shuffle_euclidean_f32_512(const unsigned char* shuffles, int codebookCount, const float* partials, float* results);
#endif

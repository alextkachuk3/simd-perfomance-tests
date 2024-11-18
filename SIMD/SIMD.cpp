#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <new>
#include <immintrin.h>

template <typename Func, typename... Args>
float measure(Func func, Args... args)
{
	int iterations = 50;
	float total_time = 0;
	for (size_t i = 0; i < iterations; i++)
	{
		auto start = std::chrono::high_resolution_clock::now();
		func(args...);
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float, std::milli> duration = end - start;
		total_time += duration.count();
	}
	return total_time / iterations;
}

template<typename T>
void default_add(T* a, T* b, T* c, size_t size)
{
	for (size_t i = 0; i < size; i++)
		c[i] = a[i] + b[i];
}

template<typename T>
void default_mul(T* a, T* b, T* c, size_t size)
{
	for (size_t i = 0; i < size; i++)
		c[i] = a[i] * b[i];
}

template<typename T>
void default_div(T* a, T* b, T* c, size_t size)
{
	for (size_t i = 0; i < size; i++)
		c[i] = a[i] / b[i];
}

template<typename T>
void default_cos(T* a, T* c, size_t size)
{
	for (size_t i = 0; i < size; i++)
		c[i] = cos(a[i]);
}

void fma_add_default(float* a, float* b, float* c, size_t size)
{
	for (size_t i = 0; i < size; i++)
		c[i] = a[i] * b[i] + c[i];
}

void avx_add_float(float* a, float* b, float* c, size_t size)
{
	size_t i = 0;
	for (; i < size; i += 8)
	{
		__m256 va = _mm256_load_ps(&a[i]);
		__m256 vb = _mm256_load_ps(&b[i]);
		__m256 vc = _mm256_add_ps(va, vb);
		_mm256_store_ps(&c[i], vc);
	}

	for (; i < size; i++)
		c[i] = a[i] + b[i];
}

void avx_mul_float(float* a, float* b, float* c, size_t size)
{
	size_t i = 0;

	for (; i < size; i += 8)
	{
		__m256 va = _mm256_load_ps(&a[i]);
		__m256 vb = _mm256_load_ps(&b[i]);
		__m256 vc = _mm256_mul_ps(va, vb);
		_mm256_store_ps(&c[i], vc);
	}

	for (; i < size; i++)
		c[i] = a[i] * b[i];
}

void avx_div_float(float* a, float* b, float* c, size_t size)
{
	size_t i = 0;

	for (; i < size; i += 8)
	{
		__m256 va = _mm256_load_ps(&a[i]);
		__m256 vb = _mm256_load_ps(&b[i]);
		__m256 vc = _mm256_div_ps(va, vb);
		_mm256_store_ps(&c[i], vc);
	}

	for (; i < size; i++)
		c[i] = a[i] / b[i];
}

void avx_cos_float(float* a, float* c, size_t size)
{
	size_t i = 0;

	for (; i < size; i += 8)
	{
		__m256 va = _mm256_load_ps(&a[i]);
		__m256 vc = _mm256_cos_ps(va);
		_mm256_store_ps(&c[i], vc);
	}

	for (; i < size; i++)
		c[i] = cos(a[i]);
}

void avx_fma_add(float* a, float* b, float* c, size_t size)
{
	size_t i = 0;
	for (; i + 7 < size; i += 8)
	{
		__m256 va = _mm256_load_ps(&a[i]);
		__m256 vb = _mm256_load_ps(&b[i]);
		__m256 vc = _mm256_load_ps(&c[i]);
		vc = _mm256_fmadd_ps(va, vb, vc);
		_mm256_store_ps(&c[i], vc);
	}
	for (; i < size; i++)
		c[i] = a[i] * b[i] + c[i];
}

void avx2_add_int(int* a, int* b, int* c, size_t size)
{
	size_t i = 0;
	for (; i + 7 < size; i += 8)
	{
		__m256i va = _mm256_load_si256((__m256i*) & a[i]);
		__m256i vb = _mm256_load_si256((__m256i*) & b[i]);
		__m256i vc = _mm256_add_epi32(va, vb);
		_mm256_store_si256((__m256i*) & c[i], vc);
	}
	for (; i < size; i++)
		c[i] = a[i] + b[i];
}

void avx2_mul_int(int* a, int* b, int* c, size_t size)
{
	size_t i = 0;
	for (; i + 7 < size; i += 8)
	{
		__m256i va = _mm256_load_si256((__m256i*) & a[i]);
		__m256i vb = _mm256_load_si256((__m256i*) & b[i]);
		__m256i vc = _mm256_mullo_epi32(va, vb);
		_mm256_store_si256((__m256i*) & c[i], vc);
	}
	for (; i < size; i++)
		c[i] = a[i] * b[i];
}

void sse_add_float(float* a, float* b, float* c, size_t size)
{
	size_t i = 0;
	for (; i + 3 < size; i += 4)
	{
		__m128 va = _mm_load_ps(&a[i]);
		__m128 vb = _mm_load_ps(&b[i]);
		__m128 vc = _mm_add_ps(va, vb);
		_mm_store_ps(&c[i], vc);
	}
	for (; i < size; i++)
		c[i] = a[i] + b[i];
}

void sse_mul_float(float* a, float* b, float* c, size_t size)
{
	size_t i = 0;
	for (; i + 3 < size; i += 4)
	{
		__m128 va = _mm_load_ps(&a[i]);
		__m128 vb = _mm_load_ps(&b[i]);
		__m128 vc = _mm_mul_ps(va, vb);
		_mm_store_ps(&c[i], vc);
	}
	for (; i < size; i++)
		c[i] = a[i] * b[i];
}

void sse_div_float(float* a, float* b, float* c, size_t size)
{
	size_t i = 0;
	for (; i + 3 < size; i += 4)
	{
		__m128 va = _mm_load_ps(&a[i]);
		__m128 vb = _mm_load_ps(&b[i]);
		__m128 vc = _mm_div_ps(va, vb);
		_mm_store_ps(&c[i], vc);
	}
	for (; i < size; i++)
		c[i] = a[i] / b[i];
}

void sse_add_int(int* a, int* b, int* c, size_t size)
{
	size_t i = 0;
	for (; i + 3 < size; i += 4)
	{
		__m128i va = _mm_load_si128((__m128i*) & a[i]);
		__m128i vb = _mm_load_si128((__m128i*) & b[i]);
		__m128i vc = _mm_add_epi32(va, vb);
		_mm_store_si128((__m128i*) & c[i], vc);
	}
	for (; i < size; i++)
		c[i] = a[i] + b[i];
}

void sse_add_double(double* a, double* b, double* c, size_t size)
{
	size_t i = 0;
	for (; i + 1 < size; i += 2)
	{
		__m128d va = _mm_load_pd(&a[i]);
		__m128d vb = _mm_load_pd(&b[i]);
		__m128d vc = _mm_add_pd(va, vb);
		_mm_store_pd(&c[i], vc);
	}
	for (; i < size; i++)
		c[i] = a[i] + b[i];
}

void sse_mul_double(double* a, double* b, double* c, size_t size)
{
	size_t i = 0;
	for (; i + 1 < size; i += 2)
	{
		__m128d va = _mm_load_pd(&a[i]);
		__m128d vb = _mm_load_pd(&b[i]);
		__m128d vc = _mm_mul_pd(va, vb);
		_mm_store_pd(&c[i], vc);
	}
	for (; i < size; i++)
		c[i] = a[i] * b[i];
}

void sse_div_double(double* a, double* b, double* c, size_t size)
{
	size_t i = 0;
	for (; i + 1 < size; i += 2)
	{
		__m128d va = _mm_load_pd(&a[i]);
		__m128d vb = _mm_load_pd(&b[i]);
		__m128d vc = _mm_div_pd(va, vb);
		_mm_store_pd(&c[i], vc);
	}
	for (; i < size; i++)
		c[i] = a[i] / b[i];
}


void simd_test()
{
	auto print_row = [](const char* op, const char* type, float scalar, float simd)
		{
			std::cout << std::setw(25) << op << std::setw(15) << type << std::setw(20) << scalar
				<< std::setw(20) << simd << std::setw(20) << (scalar / simd) << std::endl;
		};

	std::cout << std::setw(25) << "Operation" << std::setw(15) << "Type" << std::setw(20) << "Scalar (ms)"
		<< std::setw(20) << "SIMD (ms)" << std::setw(20) << "Speedup" << std::endl;
	

	std::vector<size_t> sizes = { 20000, 200000, 2000000, 20000000 };

	for (size_t size : sizes)
	{
		std::cout << size << " elements" << std::endl
			<< std::string(100, '-') << std::endl;

		{
			float* a = new float[size];
			float* b = new float[size];
			float* c = new float[size];

			for (size_t i = 0; i < size; i++)
			{
				a[i] = i + 1.0f;
				b[i] = i + 2.0f;
				c[i] = 0.0f;
			}

			float default_add_time = measure(default_add<float>, a, b, c, size);
			float avx_add_time = measure(avx_add_float, a, b, c, size);
			print_row("Addition(AVX)", "float", default_add_time, avx_add_time);

			float default_mul_time = measure(default_mul<float>, a, b, c, size);
			float avx_mul_time = measure(avx_mul_float, a, b, c, size);
			print_row("Multiplication(AVX)", "float", default_mul_time, avx_mul_time);

			float default_div_time = measure(default_div<float>, a, b, c, size);
			float avx_div_time = measure(avx_div_float, a, b, c, size);
			print_row("Division(AVX)", "float", default_div_time, avx_div_time);

			float default_cos_time = measure(default_cos<float>, a, c, size);
			float avx_cos_time = measure(avx_cos_float, a, c, size);
			print_row("Cos(AVX)", "float", default_cos_time, avx_cos_time);

			float default_fma_add_time = measure(fma_add_default, a, b, c, size);
			float avx_fma_add_time = measure(avx_fma_add, a, b, c, size);
			print_row("FMA addition(AVX)", "float", default_fma_add_time, avx_fma_add_time);

			delete[] a;
			delete[] b;
			delete[] c;
		}

		{
			int* a = new int[size];
			int* b = new int[size];
			int* c = new int[size];

			for (size_t i = 0; i < size; i++)
			{
				a[i] = i + 1;
				b[i] = i + 2;
				c[i] = 0;
			}

			float avx2_add_time = measure(avx2_add_int, a, b, c, size);
			float default_add_time = measure(default_add<int>, a, b, c, size);
			print_row("Addition(AVX2)", "int", avx2_add_time, default_add_time);

			float avx2_mult_time = measure(avx2_mul_int, a, b, c, size);
			float default_mult_time = measure(default_mul<int>, a, b, c, size);
			print_row("Multiplication(AVX2)", "int", avx2_mult_time, default_mult_time);

			delete[] a;
			delete[] b;
			delete[] c;
		}

		{
			float* a = new float[size];
			float* b = new float[size];
			float* c = new float[size];

			for (size_t i = 0; i < size; i++)
			{
				a[i] = i + 1.0f;
				b[i] = i + 2.0f;
				c[i] = 0.0f;
			}

			float default_add_time = measure(default_add<float>, a, b, c, size);
			float sse_add_time = measure(sse_add_float, a, b, c, size);
			print_row("Addition(SSE)", "float", default_add_time, sse_add_time);

			float default_mul_time = measure(default_mul<float>, a, b, c, size);
			float sse_mul_time = measure(sse_mul_float, a, b, c, size);
			print_row("Multiplication(SSE)", "float", default_mul_time, sse_mul_time);

			float default_div_time = measure(default_div<float>, a, b, c, size);
			float sse_div_time = measure(sse_div_float, a, b, c, size);
			print_row("Division(SSE)", "float", default_div_time, sse_div_time);

			delete[] a;
			delete[] b;
			delete[] c;
		}

		{
			double* a = new double[size];
			double* b = new double[size];
			double* c = new double[size];

			for (size_t i = 0; i < size; i++)
			{
				a[i] = i + 1.0;
				b[i] = i + 2.0;
				c[i] = 0.0;
			}

			float default_add_time = measure(default_add<double>, a, b, c, size);
			float sse_add_time = measure(sse_add_double, a, b, c, size);
			print_row("Addition(SSE)", "double", default_add_time, sse_add_time);

			float default_mul_time = measure(default_mul<double>, a, b, c, size);
			float sse_mul_time = measure(sse_mul_double, a, b, c, size);
			print_row("Multiplication(SSE)", "double", default_mul_time, sse_mul_time);

			float default_div_time = measure(default_div<double>, a, b, c, size);
			float sse_div_time = measure(sse_div_double, a, b, c, size);
			print_row("Division(SSE)", "double", default_div_time, sse_div_time);

			delete[] a;
			delete[] b;
			delete[] c;
		}
	}	
}

int main()
{
	std::cout << "SIMD Performance Test" << std::endl;
	simd_test();
	return 0;
}

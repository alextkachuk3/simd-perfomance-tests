#include <iostream>
#include <chrono>
#include <iomanip>
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
	{
		c[i] = a[i] + b[i];
	}
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
	{
		c[i] = a[i] * b[i];
	}
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
	{
		c[i] = a[i] / b[i];
	}
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
	{
		c[i] = cos(a[i]);
	}
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
	std::cout << std::string(100, '-') << std::endl;

	size_t size = 20000000;

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

	float default_mul_time = measure(default_mul<float>, a, b, c, size);
	float avx_mul_time = measure(avx_mul_float, a, b, c, size);

	float default_div_time = measure(default_div<float>, a, b, c, size);
	float avx_div_time = measure(avx_div_float, a, b, c, size);	

	float default_cos_time = measure(default_cos<float>, a, c, size);
	float avx_cos_time = measure(avx_cos_float, a, c, size);

	print_row("Addition(AVX)", "float", default_add_time, avx_add_time);
	print_row("Multiplication(AVX)", "float", default_mul_time, avx_mul_time);
	print_row("Division(AVX)", "float", default_div_time, avx_div_time);
	print_row("Cos(AVX)", "float", default_cos_time, avx_cos_time);

	delete[] a;
	delete[] b;
	delete[] c;
}

int main()
{
	std::cout << "SIMD Performance Test" << std::endl;
	simd_test();
	return 0;
}

#include "core.hpp"

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>

#define LBFGS_TEST_CORRECTNESS() 1

inline auto make_random_buffer(size_t const size)
{
    struct Deleter {
        auto operator()(float* p) noexcept -> void { std::free(p); }
    };
    auto* p = reinterpret_cast<float*>(
        std::aligned_alloc(64UL, sizeof(float) * size));
    if (p == nullptr) { throw std::bad_alloc{}; }
    auto buffer = std::unique_ptr<float[], Deleter>{p};

    std::mt19937 gen{std::random_device{}()};
    std::generate(buffer.get(), buffer.get() + size, [&gen]() {
        return std::uniform_real_distribution<float>{-1.0, 1.0}(gen);
    });
    return buffer;
}

static void bm_blas_dot(benchmark::State& state)
{
    auto const n     = 1000029UL;
    auto       x_raw = make_random_buffer(n);
    auto       y_raw = make_random_buffer(n);
    auto       x     = gsl::span<float const>{x_raw.get(), n};
    auto       y     = gsl::span<float const>{y_raw.get(), n};
#if 0
    std::printf("%.10e vs. %.10e vs. %.10e\n", blas_dot(x, y), slow_dot(x, y),
                custom_dot(x, y));
#    if 0
    std::fprintf(stderr, "------------------------------\n");
    for (auto i = size_t{0}; i < n; ++i) {
        std::fprintf(stderr, "%.20e\t%.20e\n", x[i], y[i]);
    }
    std::fprintf(stderr, "------------------------------\n");
#    endif
#endif
    for (auto _ : state) {
        blas_dot(x, y);
    }
}

static void bm_custom_dot(benchmark::State& state)
{
    auto const n     = 1000029UL;
    auto       x_raw = make_random_buffer(n);
    auto       y_raw = make_random_buffer(n);
    auto       x     = gsl::span<float const>{x_raw.get(), n};
    auto       y     = gsl::span<float const>{y_raw.get(), n};
    for (auto _ : state) {
        custom_dot(x, y);
    }
}

static void bm_blas_nrm2(benchmark::State& state)
{
#if LBFGS_TEST_CORRECTNESS()
    static bool first_time = true;
#endif
    auto const n     = 1000029UL;
    auto       x_raw = make_random_buffer(n);
    auto       x     = gsl::span<float const>{x_raw.get(), n};

#if LBFGS_TEST_CORRECTNESS()
    if (first_time) {
        std::printf("%.15e vs. %.15e\n", blas_nrm2(x), custom_nrm2(x));
        auto* fp = std::fopen("test_nrm2.txt", "w");
        if (!fp) { throw std::runtime_error{"file opening failed"}; }
        for (auto i = size_t{0}; i < n; ++i) {
            std::fprintf(fp, "%.20e\n", x[i]);
        }
        std::fclose(fp);
        first_time = false;
    }
#endif
    for (auto _ : state) {
        blas_nrm2(x);
    }
}

static void bm_custom_nrm2(benchmark::State& state)
{
    auto const n     = 1000029UL;
    auto       x_raw = make_random_buffer(n);
    auto       x     = gsl::span<float const>{x_raw.get(), n};
    for (auto _ : state) {
        custom_nrm2(x);
    }
}

static void bm_blas_scal(benchmark::State& state)
{
    auto const n     = 1000029UL;
    auto       x_raw = make_random_buffer(n);
    auto       x     = gsl::span<float>{x_raw.get(), n};
    for (auto _ : state) {
        blas_scal(0.123532f, x);
    }
}

static void bm_custom_scal(benchmark::State& state)
{
    auto const n     = 1000029UL;
    auto       x_raw = make_random_buffer(n);
    auto       x     = gsl::span<float>{x_raw.get(), n};
    for (auto _ : state) {
        custom_scal(0.123532f, x);
    }
}

static void bm_blas_axpy(benchmark::State& state)
{
    auto const n     = 10000029UL;
    auto       x_raw = make_random_buffer(n);
    auto       y_raw = make_random_buffer(n);
    auto       x     = gsl::span<float const>{x_raw.get(), n};
    auto       y     = gsl::span<float>{y_raw.get(), n};
    for (auto _ : state) {
        blas_axpy(-1.001123f, x, y);
    }
}

static void bm_custom_axpy(benchmark::State& state)
{
    auto const n     = 10000029UL;
    auto       x_raw = make_random_buffer(n);
    auto       y_raw = make_random_buffer(n);
    auto       x     = gsl::span<float const>{x_raw.get(), n};
    auto       y     = gsl::span<float>{y_raw.get(), n};
    for (auto _ : state) {
        custom_axpy(-1.001123f, x, y);
    }
}

BENCHMARK(bm_blas_dot);
BENCHMARK(bm_custom_dot);
BENCHMARK(bm_blas_nrm2);
BENCHMARK(bm_custom_nrm2);
BENCHMARK(bm_blas_scal);
BENCHMARK(bm_custom_scal);
BENCHMARK(bm_blas_axpy);
BENCHMARK(bm_custom_axpy);

BENCHMARK_MAIN();

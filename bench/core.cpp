#include "core.hpp"
#include "lbfgs.hpp"

#include <cblas.h>
#include <immintrin.h>

template <class T> struct get_blas_int_type;

template <class T>
struct get_blas_int_type<float (*)(T, float const*, T, float const*, T)> {
    using type = T;
};

using blas_int = typename get_blas_int_type<decltype(&cblas_sdot)>::type;

LBFGS_EXPORT auto blas_dot(gsl::span<float const> a,
                           gsl::span<float const> b) noexcept -> double
{
    LBFGS_ASSERT(a.size() == b.size(), "incompatible dimensions");
    LBFGS_ASSERT(
        a.size() <= static_cast<size_t>(std::numeric_limits<blas_int>::max()),
        "integer overflow");
    return cblas_dsdot(static_cast<blas_int>(a.size()), a.data(), 1, b.data(),
                       1);
}

LBFGS_EXPORT auto blas_nrm2(gsl::span<float const> x) noexcept -> double
{
    LBFGS_ASSERT(
        x.size() <= static_cast<size_t>(std::numeric_limits<blas_int>::max()),
        "integer overflow");
    return static_cast<double>(
        cblas_snrm2(static_cast<blas_int>(x.size()), x.data(), 1));
}

LBFGS_EXPORT auto blas_scal(float const a, gsl::span<float> x) noexcept -> void
{
    LBFGS_ASSERT(
        x.size() <= static_cast<size_t>(std::numeric_limits<blas_int>::max()),
        "integer overflow");
    cblas_sscal(static_cast<blas_int>(x.size()), a, x.data(), 1);
}

LBFGS_EXPORT auto blas_axpy(float const a, gsl::span<float const> x,
                            gsl::span<float> y) noexcept -> void
{
    LBFGS_ASSERT(x.size() == y.size(), "incompatible dimensions");
    LBFGS_ASSERT(
        x.size() <= static_cast<size_t>(std::numeric_limits<blas_int>::max()),
        "integer overflow");
    cblas_saxpy(static_cast<blas_int>(x.size()), a, x.data(), 1, y.data(), 1);
}

LBFGS_EXPORT auto custom_dot(gsl::span<float const> a,
                             gsl::span<float const> b) noexcept -> double
{
    return ::LBFGS_NAMESPACE::detail::dot(a, b);
}

LBFGS_EXPORT auto custom_nrm2(gsl::span<float const> a) noexcept -> double
{
    return ::LBFGS_NAMESPACE::detail::nrm2(a);
}

LBFGS_EXPORT auto custom_scal(float const c, gsl::span<float> a) noexcept
    -> void
{
    return ::LBFGS_NAMESPACE::detail::scal(c, a);
}

LBFGS_EXPORT auto custom_axpy(float a, gsl::span<float const> x,
                              gsl::span<float> y) noexcept -> void
{
    return ::LBFGS_NAMESPACE::detail::axpy(a, x, y);
}

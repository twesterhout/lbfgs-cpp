// Copyright (c) 2019, Tom Westerhout
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// vim: foldenable foldmethod=marker
#include "lbfgs/lbfgs.hpp"
#if defined(LBFGS_USE_BLAS)
#    include <cblas.h>
#else
#    include <immintrin.h>
#endif

LBFGS_NAMESPACE_BEGIN

// ============================= Error codes =============================== {{{
namespace { // anonymous namespace
struct lbfgs_error_category : public std::error_category {
    constexpr lbfgs_error_category() noexcept = default;

    lbfgs_error_category(lbfgs_error_category const&) = delete;
    lbfgs_error_category(lbfgs_error_category&&)      = delete;
    auto operator                =(lbfgs_error_category const&)
        -> lbfgs_error_category& = delete;
    auto operator=(lbfgs_error_category &&) -> lbfgs_error_category& = delete;

    ~lbfgs_error_category() override = default;

    [[nodiscard]] auto        name() const noexcept -> char const* override;
    [[nodiscard]] auto        message(int value) const -> std::string override;
    [[nodiscard]] static auto instance() noexcept -> std::error_category const&;
};

auto lbfgs_error_category::name() const noexcept -> char const*
{
    return "lbfgs category";
}

auto lbfgs_error_category::message(int const value) const -> std::string
{
    switch (static_cast<status_t>(value)) {
    case status_t::success: return "no error";
    case status_t::out_of_memory: return "out of memory";
    case status_t::invalid_storage_size: return "invalid m (storage size)";
    case status_t::invalid_epsilon: return "invalid epsilon";
    case status_t::invalid_delta: return "invalid delta";
    case status_t::too_many_iterations: return "too many iterations";
    case status_t::invalid_argument: return "received an invalid argument";
    case status_t::rounding_errors_prevent_progress:
        return "rounding errors prevent further progress";
    case status_t::maximum_step_reached: return "line search reached αₘₐₓ";
    case status_t::minimum_step_reached: return "line search reached αₘᵢₙ";
    case status_t::too_many_function_evaluations:
        return "too many function evaluations";
    case status_t::interval_too_small:
        return "line search interval has shrunk below threshold";
    case status_t::invalid_interval_tolerance: return "invalid interval width";
    case status_t::invalid_function_tolerance: return "invalid parameter μ";
    case status_t::invalid_gradient_tolerance: return "invalid parameter η";
    case status_t::invalid_step_bounds: return "invalid interval [αₘᵢₙ, αₘₐₓ]";
#if defined(LBFGS_CLANG)
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wcovered-switch-default"
#endif
    // NOTE: We do want the default case, because the user could have constructed an
    // invalid error code using our category
    // NOLINTNEXTLINE
    default: return "(unrecognised error)";
#if defined(LBFGS_CLANG)
#    pragma clang diagnostic pop
#endif
    } // end switch
}

auto lbfgs_error_category::instance() noexcept -> std::error_category const&
{
#if defined(LBFGS_CLANG)
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wexit-time-destructors"
#endif
    static lbfgs_error_category c; // NOLINT
#if defined(LBFGS_CLANG)
#    pragma clang diagnostic pop
#endif
    return c;
}
} // namespace

LBFGS_EXPORT auto make_error_code(status_t const e) noexcept -> std::error_code
{
    return {static_cast<int>(e), lbfgs_error_category::instance()};
}
// ============================= Error codes =============================== }}}

namespace detail {

[[noreturn]] LBFGS_EXPORT auto assert_fail(char const* expr, char const* file,
                                           unsigned line, char const* function,
                                           char const* msg) noexcept -> void
{
    // NOLINTNEXTLINE
    std::fprintf(stderr,
                 LBFGS_BUG_MESSAGE
                 "\n\x1b[1m\x1b[91mAssertion failed\x1b[0m at %s:%u: %s: "
                 "\"\x1b[1m\x1b[97m%s\x1b[0m\" evaluated to false: "
                 "\x1b[1m\x1b[97m%s\x1b[0m\n",
                 file, line, function, expr, msg);
    std::terminate();
}

// ============================= Line search =============================== {{{
namespace {
    /// \brief Case 1 on p. 299 of [1].
    ///
    /// \return `(αₜ⁺, bracketed, bound)` where `αₜ⁺` is the trial value in the
    ///         new search interval `I⁺`.
    inline auto case_1(ls_state_t const& state) noexcept
        -> std::tuple<double, bool, bool>
    {
        auto const cubic = detail::minimise_cubic_interpolation(
            /*a=*/state.x.alpha, /*f_a=*/state.x.func, /*df_a=*/state.x.grad,
            /*b=*/state.t.alpha, /*f_b=*/state.t.func, /*df_b=*/state.t.grad);
        auto const quadratic = detail::minimise_quadratic_interpolation(
            /*a=*/state.x.alpha, /*f_a=*/state.x.func, /*df_a=*/state.x.grad,
            /*b=*/state.t.alpha, /*f_b=*/state.t.func);
        auto const alpha = std::abs(cubic - state.x.alpha)
                                   < std::abs(quadratic - state.x.alpha)
                               ? cubic
                               : cubic + 0.5 * (quadratic - cubic);
        LBFGS_TRACE("case_1: α_c=%.5e, α_q=%.5e -> α=%.5e\n", cubic, // NOLINT
                    quadratic,                                       // NOLINT
                    alpha);                                          // NOLINT
        return {alpha, /*bracketed=*/true, /*bound=*/true};
    }

    /// \brief Case 2 on p. 299 of [1].
    ///
    /// \return `(αₜ⁺, bracketed, bound)` where `αₜ⁺` is the trial value in the
    ///         new search interval `I⁺`.
    inline auto case_2(ls_state_t const& state) noexcept
        -> std::tuple<double, bool, bool>
    {
        auto const cubic = detail::minimise_cubic_interpolation(
            /*a=*/state.x.alpha, /*f_a=*/state.x.func,
            /*df_a=*/state.x.grad,
            /*b=*/state.t.alpha, /*f_b=*/state.t.func,
            /*df_b=*/state.t.grad);
        auto const secant = detail::minimise_quadratic_interpolation(
            /*a=*/state.x.alpha, /*df_a=*/state.x.grad,
            /*b=*/state.t.alpha, /*df_b=*/state.t.grad);
        auto const alpha =
            std::abs(cubic - state.t.alpha) >= std::abs(secant - state.t.alpha)
                ? cubic
                : secant;
        LBFGS_TRACE("case_2: α_c=%.5e, α_s=%.5e -> α=%.5e\n", cubic, // NOLINT
                    secant,                                          // NOLINT
                    alpha);                                          // NOLINT
        return {alpha, /*bracketed=*/true, /*bound=*/false};
    }

    inline auto case_3(ls_state_t const& state) noexcept
        -> std::tuple<double, bool, bool>
    {
        auto const cubic = detail::minimise_cubic_interpolation(
            /*a=*/state.x.alpha, /*f_a=*/state.x.func,
            /*df_a=*/state.x.grad,
            /*b=*/state.t.alpha, /*f_b=*/state.t.func,
            /*df_b=*/state.t.grad);
        auto const secant = detail::minimise_quadratic_interpolation(
            /*a=*/state.x.alpha, /*df_a=*/state.x.grad,
            /*b=*/state.t.alpha, /*df_b=*/state.t.grad);
        if (!std::isinf(cubic)
            && (cubic - state.t.alpha) * (state.t.alpha - state.x.alpha)
                   >= 0.0) {
            static_assert(std::is_same_v<decltype(state.bracketed), bool>);
            auto const condition = state.bracketed
                                   == (std::abs(cubic - state.t.alpha)
                                       < std::abs(secant - state.t.alpha));
            auto result = std::tuple{condition ? cubic : secant,
                                     /*bracketed=*/state.bracketed,
                                     /*bound=*/true};
            LBFGS_TRACE(                                        // NOLINT
                "case_3 (true): α_l=%.5e, α_t=%.5e, α_c=%.5e, " // NOLINT
                "α_s=%.5e -> α=%.5e\n",                         // NOLINT
                state.x.alpha, state.t.alpha, cubic, secant,    // NOLINT
                std::get<0>(result));                           // NOLINT
            return result;
        }
        auto result =
            std::tuple{secant, /*bracketed=*/state.bracketed, /*bound=*/true};
        LBFGS_TRACE(                                         // NOLINT
            "case_3 (false): α_l=%.5e, α_t=%.5e, α_c=%.5e, " // NOLINT
            "α_s=%.5e -> α=%.5e\n",                          // NOLINT
            state.x.alpha, state.t.alpha, cubic, secant,     // NOLINT
            std::get<0>(result));                            // NOLINT
        return result;
    }

    inline auto case_4(ls_state_t const& state) noexcept
        -> std::tuple<double, bool, bool>
    {
        auto const alpha =
            state.bracketed
                ? detail::minimise_cubic_interpolation(
                    /*a=*/state.t.alpha, /*f_a=*/state.t.func,
                    /*df_a=*/state.t.grad,
                    /*b=*/state.y.alpha, /*f_b=*/state.y.func,
                    /*df_b=*/state.y.grad)
                : std::copysign(std::numeric_limits<float>::infinity(),
                                state.t.alpha - state.x.alpha);
        LBFGS_TRACE("case_4: α=%f\n", // NOLINT
                    alpha);           // NOLINT
        return {alpha, /*bracketed=*/state.bracketed, /*bound=*/false};
    }

    inline auto handle_cases(ls_state_t const& state) noexcept
        -> std::tuple<double, bool, bool>
    {
        if (state.t.func > state.x.func) { return case_1(state); }
        if (state.x.grad * state.t.grad < 0.0) { return case_2(state); }
        // NOTE(twesterhout): The paper uses `<=` here!
        if (std::abs(state.t.grad) < std::abs(state.x.grad)) {
            return case_3(state);
        }
        return case_4(state);
    }
} // namespace

LBFGS_EXPORT auto update_trial_value_and_interval(ls_state_t& state) noexcept
    -> void
{
    // Check the input parameters for errors.
    LBFGS_ASSERT(
        !state.bracketed
            || (std::min(state.x.alpha, state.y.alpha) < state.t.alpha
                && state.t.alpha < std::max(state.x.alpha, state.y.alpha)),
        "αₜ ∉ I");
    LBFGS_ASSERT(state.x.grad * (state.t.alpha - state.x.alpha) < 0.0,
                 "wrong search direction");
    bool   bound;
    double alpha;
    std::tie(alpha, state.bracketed, bound) = handle_cases(state);

    if (state.t.func > state.x.func) { state.y = state.t; }
    else {
        if (state.x.grad * state.t.grad <= 0.0) { state.y = state.x; }
        state.x = state.t;
    }
    LBFGS_TRACE("cstep: new α_l=%f, α_u=%f\n", state.x.alpha, // NOLINT
                state.y.alpha);                               // NOLINT
    alpha = std::clamp(alpha, state.interval.min(), state.interval.max());
    if (state.bracketed && bound) {
        auto const middle =
            state.x.alpha + 0.66 * (state.y.alpha - state.x.alpha);
        LBFGS_TRACE("cstep: bracketed && bound: α=%f, middle=%f\n", // NOLINT
                    alpha,                                          // NOLINT
                    middle);                                        // NOLINT
        alpha = (state.x.alpha < state.y.alpha) ? std::min(middle, alpha)
                                                : std::max(middle, alpha);
    }
    state.t.alpha  = alpha;
    state.t.func   = std::numeric_limits<double>::quiet_NaN();
    state.t.grad   = std::numeric_limits<double>::quiet_NaN();
    state.interval = interval_t{state.x.alpha, state.y.alpha, state.t.alpha,
                                state.bracketed};
}
} // namespace detail
// ========================================================================= }}}

namespace detail {
// ================================= BLAS ================================== {{{
#if defined(LBFGS_USE_BLAS)
// A hacky way of determining the integral type BLAS uses for sizes and
// increments: we pattern match on the signature of `cblas_sdot`.
template <class T> struct get_blas_int_type;

template <class T>
struct get_blas_int_type<float (*)(T, float const*, T, float const*, T)> {
    using type = T;
};

using blas_int = typename get_blas_int_type<decltype(&cblas_sdot)>::type;

LBFGS_EXPORT auto dot(gsl::span<float const> a,
                      gsl::span<float const> b) noexcept -> double
{
    LBFGS_ASSERT(a.size() == b.size(), "incompatible dimensions");
    LBFGS_ASSERT(
        a.size() <= static_cast<size_t>(std::numeric_limits<blas_int>::max()),
        "integer overflow");
    return cblas_dsdot(static_cast<blas_int>(a.size()), a.data(), 1, b.data(),
                       1);
}

LBFGS_EXPORT auto nrm2(gsl::span<float const> x) noexcept -> double
{
    LBFGS_ASSERT(
        x.size() <= static_cast<size_t>(std::numeric_limits<blas_int>::max()),
        "integer overflow");
#    if defined(LBFGS_CLANG)
#        pragma clang diagnostic push
#        pragma clang diagnostic ignored "-Wdouble-promotion"
#    endif
    return cblas_snrm2(static_cast<blas_int>(x.size()), x.data(), 1);
#    if defined(LBFGS_CLANG)
#        pragma clang diagnostic pop
#    endif
}

LBFGS_EXPORT auto scal(float const a, gsl::span<float> x) noexcept -> void
{
    LBFGS_ASSERT(
        x.size() <= static_cast<size_t>(std::numeric_limits<blas_int>::max()),
        "integer overflow");
    cblas_sscal(static_cast<blas_int>(x.size()), a, x.data(), 1);
}

LBFGS_EXPORT auto negative_copy(gsl::span<float const> const src,
                                gsl::span<float> const dst) noexcept -> void
{
    LBFGS_ASSERT(src.size() == dst.size(), "incompatible dimensions");
    for (auto i = size_t{0}; i < src.size(); ++i) {
        dst[i] = -src[i];
    }
}

LBFGS_EXPORT auto axpy(float const a, gsl::span<float const> x,
                       gsl::span<float> y) noexcept -> void
{
    LBFGS_ASSERT(x.size() == y.size(), "incompatible dimensions");
    LBFGS_ASSERT(
        x.size() <= static_cast<size_t>(std::numeric_limits<blas_int>::max()),
        "integer overflow");
    cblas_saxpy(static_cast<blas_int>(x.size()), a, x.data(), 1, y.data(), 1);
}

LBFGS_EXPORT auto axpy(float const a, gsl::span<float const> x,
                       gsl::span<float const> y, gsl::span<float> out) noexcept
    -> void
{
    LBFGS_ASSERT(x.size() == y.size() && y.size() == out.size(),
                 "incompatible dimensions");
    std::memcpy(out.data(), y.data(), out.size() * sizeof(float));
    axpy(a, x, out);
}

#else

namespace {
    LBFGS_FORCEINLINE auto hsum(__m256d v) noexcept -> double
    {
        auto vlow    = _mm256_castpd256_pd128(v);
        auto vhigh   = _mm256_extractf128_pd(v, 1);
        vlow         = _mm_add_pd(vlow, vhigh);
        auto undef   = _mm_undefined_ps();
        auto shuftmp = _mm_movehl_ps(undef, _mm_castpd_ps(vlow));
        auto shuf    = _mm_castps_pd(shuftmp);
        return _mm_cvtsd_f64(_mm_add_sd(vlow, shuf));
    }

    /// Dot product of 32 floats using `double`s for accumulation.
    LBFGS_FORCEINLINE auto dot_kernel_32(float const* LBFGS_RESTRICT x,
                                         float const* LBFGS_RESTRICT y) noexcept
        -> __m256d
    {
        __m256  x0, x1, x2, x3;
        __m256  y0, y1, y2, y3;
        __m256d a0, a1, a2, a3, a4, a5, a6, a7;
        __m256d b0, b1, b2, b3, b4, b5, b6, b7;

        x0 = _mm256_load_ps(x);
        x1 = _mm256_load_ps(x + 8);
        x2 = _mm256_load_ps(x + 16);
        x3 = _mm256_load_ps(x + 24);
        y0 = _mm256_load_ps(y);
        y1 = _mm256_load_ps(y + 8);
        y2 = _mm256_load_ps(y + 16);
        y3 = _mm256_load_ps(y + 24);

        a1 = _mm256_cvtps_pd(_mm256_extractf128_ps(x0, 1));
        a0 = _mm256_cvtps_pd(_mm256_extractf128_ps(x0, 0));
        a3 = _mm256_cvtps_pd(_mm256_extractf128_ps(x1, 1));
        a2 = _mm256_cvtps_pd(_mm256_extractf128_ps(x1, 0));
        a5 = _mm256_cvtps_pd(_mm256_extractf128_ps(x2, 1));
        a4 = _mm256_cvtps_pd(_mm256_extractf128_ps(x2, 0));
        a7 = _mm256_cvtps_pd(_mm256_extractf128_ps(x3, 1));
        a6 = _mm256_cvtps_pd(_mm256_extractf128_ps(x3, 0));

        b1 = _mm256_cvtps_pd(_mm256_extractf128_ps(y0, 1));
        b0 = _mm256_cvtps_pd(_mm256_extractf128_ps(y0, 0));
        b3 = _mm256_cvtps_pd(_mm256_extractf128_ps(y1, 1));
        b2 = _mm256_cvtps_pd(_mm256_extractf128_ps(y1, 0));
        b5 = _mm256_cvtps_pd(_mm256_extractf128_ps(y2, 1));
        b4 = _mm256_cvtps_pd(_mm256_extractf128_ps(y2, 0));
        b7 = _mm256_cvtps_pd(_mm256_extractf128_ps(y3, 1));
        b6 = _mm256_cvtps_pd(_mm256_extractf128_ps(y3, 0));

        a0 = _mm256_mul_pd(a0, b0);
        a1 = _mm256_mul_pd(a1, b1);
        a2 = _mm256_mul_pd(a2, b2);
        a3 = _mm256_mul_pd(a3, b3);
        a4 = _mm256_mul_pd(a4, b4);
        a5 = _mm256_mul_pd(a5, b5);
        a6 = _mm256_mul_pd(a6, b6);
        a7 = _mm256_mul_pd(a7, b7);

        a0 = _mm256_add_pd(a0, a1);
        a2 = _mm256_add_pd(a2, a3);
        a4 = _mm256_add_pd(a4, a5);
        a6 = _mm256_add_pd(a6, a7);

        a0 = _mm256_add_pd(a0, a2);
        a4 = _mm256_add_pd(a4, a6);

        return _mm256_add_pd(a0, a4);
    }

    /// Dot product of 8 floats using `double`s for accumulation.
    LBFGS_FORCEINLINE auto dot_kernel_8(float const* LBFGS_RESTRICT x,
                                        float const* LBFGS_RESTRICT y) noexcept
        -> __m256d
    {
        __m256  x0;
        __m256  y0;
        __m256d a0, a1;
        __m256d b0, b1;

        x0 = _mm256_load_ps(x);
        y0 = _mm256_load_ps(y);

        a1 = _mm256_cvtps_pd(_mm256_extractf128_ps(x0, 1));
        a0 = _mm256_cvtps_pd(_mm256_extractf128_ps(x0, 0));

        b1 = _mm256_cvtps_pd(_mm256_extractf128_ps(y0, 1));
        b0 = _mm256_cvtps_pd(_mm256_extractf128_ps(y0, 0));

        a0 = _mm256_mul_pd(a0, b0);
        a1 = _mm256_mul_pd(a1, b1);

        return _mm256_add_pd(a0, a1);
    }
} // namespace

LBFGS_EXPORT auto dot(gsl::span<float const> a,
                      gsl::span<float const> b) noexcept -> double
{
    LBFGS_ASSERT(a.size() == b.size(), "incompatible dimensions");
    LBFGS_ASSERT(
        a.size() <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
        "integer overflow");
    LBFGS_ASSERT(reinterpret_cast<std::uintptr_t>(a.data()) % 64UL == 0,
                 "a must be aligned to 64-byte boundary");
    LBFGS_ASSERT(reinterpret_cast<std::uintptr_t>(b.data()) % 64UL == 0,
                 "b must be aligned to 64-byte boundary");
    auto                 n   = static_cast<int64_t>(a.size());
    auto                 sum = _mm256_set1_pd(0.0);
    auto* LBFGS_RESTRICT x   = a.data();
    auto* LBFGS_RESTRICT y   = b.data();
    for (; n >= 32; n -= 32, x += 32, y += 32) {
        sum = _mm256_add_pd(sum, dot_kernel_32(x, y));
    }
    for (; n >= 8; n -= 8, x += 8, y += 8) {
        sum = _mm256_add_pd(sum, dot_kernel_8(x, y));
    }
    auto total = hsum(sum);
    for (; n > 0; --n, ++x, ++y) {
        total += static_cast<double>(*x) * static_cast<double>(*y);
    }
    return total;
}

namespace {
    LBFGS_FORCEINLINE auto nrm2_kernel_32(float const* x) noexcept -> __m256d
    {
        __m256  x0, x1, x2, x3;
        __m256d a0, a1, a2, a3, a4, a5, a6, a7;

        x0 = _mm256_load_ps(x);
        x1 = _mm256_load_ps(x + 8);
        x2 = _mm256_load_ps(x + 16);
        x3 = _mm256_load_ps(x + 24);

        a1 = _mm256_cvtps_pd(_mm256_extractf128_ps(x0, 1));
        a0 = _mm256_cvtps_pd(_mm256_extractf128_ps(x0, 0));
        a3 = _mm256_cvtps_pd(_mm256_extractf128_ps(x1, 1));
        a2 = _mm256_cvtps_pd(_mm256_extractf128_ps(x1, 0));
        a5 = _mm256_cvtps_pd(_mm256_extractf128_ps(x2, 1));
        a4 = _mm256_cvtps_pd(_mm256_extractf128_ps(x2, 0));
        a7 = _mm256_cvtps_pd(_mm256_extractf128_ps(x3, 1));
        a6 = _mm256_cvtps_pd(_mm256_extractf128_ps(x3, 0));

        a0 = _mm256_mul_pd(a0, a0);
        a1 = _mm256_mul_pd(a1, a1);
        a2 = _mm256_mul_pd(a2, a2);
        a3 = _mm256_mul_pd(a3, a3);
        a4 = _mm256_mul_pd(a4, a4);
        a5 = _mm256_mul_pd(a5, a5);
        a6 = _mm256_mul_pd(a6, a6);
        a7 = _mm256_mul_pd(a7, a7);

        a0 = _mm256_add_pd(a0, a1);
        a2 = _mm256_add_pd(a2, a3);
        a4 = _mm256_add_pd(a4, a5);
        a6 = _mm256_add_pd(a6, a7);

        a0 = _mm256_add_pd(a0, a2);
        a4 = _mm256_add_pd(a4, a6);

        return _mm256_add_pd(a0, a4);
    }

    LBFGS_FORCEINLINE auto nrm2_kernel_8(float const* x) noexcept -> __m256d
    {
        __m256  x0;
        __m256d a0, a1;

        x0 = _mm256_load_ps(x);

        a1 = _mm256_cvtps_pd(_mm256_extractf128_ps(x0, 1));
        a0 = _mm256_cvtps_pd(_mm256_extractf128_ps(x0, 0));

        a0 = _mm256_mul_pd(a0, a0);
        a1 = _mm256_mul_pd(a1, a1);

        return _mm256_add_pd(a0, a1);
    }
} // namespace

LBFGS_EXPORT auto nrm2(gsl::span<float const> a) noexcept -> double
{
    LBFGS_ASSERT(
        a.size() <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
        "integer overflow");
    LBFGS_ASSERT(reinterpret_cast<std::uintptr_t>(a.data()) % 64UL == 0,
                 "x must be aligned to 64-byte boundary");
    auto  n   = static_cast<int64_t>(a.size());
    auto  sum = _mm256_set1_pd(0.0);
    auto* x   = a.data();
    for (; n >= 32; n -= 32, x += 32) {
        sum = _mm256_add_pd(sum, nrm2_kernel_32(x));
    }
    for (; n >= 8; n -= 8, x += 8) {
        sum = _mm256_add_pd(sum, nrm2_kernel_8(x));
    }
    auto total = hsum(sum);
    for (; n > 0; --n, ++x) {
        auto t = static_cast<double>(*x);
        total += t * t;
    }
    return std::sqrt(total);
}

LBFGS_EXPORT auto scal(float const a, gsl::span<float> const x) noexcept -> void
{
    LBFGS_ASSERT(
        x.size() <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
        "integer overflow");
    LBFGS_ASSERT(reinterpret_cast<std::uintptr_t>(x.data()) % 64UL == 0,
                 "x must be aligned to 64-byte boundary");
    auto       n   = static_cast<int64_t>(x.size());
    auto const a_v = _mm256_set1_ps(a);
    auto*      p   = x.data();
    for (; n >= 16; n -= 16, p += 16) {
        _mm256_store_ps(p, _mm256_mul_ps(a_v, _mm256_load_ps(p)));
        _mm256_store_ps(p + 8, _mm256_mul_ps(a_v, _mm256_load_ps(p + 8)));
    }
    for (; n > 0; --n, ++p) {
        (*p) *= a;
    }
}

LBFGS_EXPORT auto negative_copy(gsl::span<float const> const src,
                                gsl::span<float> const dst) noexcept -> void
{
    LBFGS_ASSERT(src.size() == dst.size(), "incompatible dimensions");
    LBFGS_ASSERT(
        src.size() <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
        "integer overflow");
    LBFGS_ASSERT(reinterpret_cast<std::uintptr_t>(src.data()) % 64UL == 0,
                 "src must be aligned to 64-byte boundary");
    LBFGS_ASSERT(reinterpret_cast<std::uintptr_t>(dst.data()) % 64UL == 0,
                 "dst must be aligned to 64-byte boundary");
    auto                 n    = static_cast<int64_t>(src.size());
    auto const           zero = _mm256_set1_ps(0.0f);
    auto* LBFGS_RESTRICT s    = src.data();
    auto* LBFGS_RESTRICT d    = dst.data();
    for (; n >= 8; n -= 8, s += 8, d += 8) {
        _mm256_store_ps(d, _mm256_sub_ps(zero, _mm256_load_ps(s)));
    }
    for (; n > 0; --n, ++s, ++d) {
        (*d) = -(*s);
    }
}

LBFGS_EXPORT auto axpy(float const a, gsl::span<float const> x,
                       gsl::span<float> y) noexcept -> void
{
    LBFGS_ASSERT(x.size() == y.size(), "incompatible dimensions");
    LBFGS_ASSERT(
        x.size() <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
        "integer overflow");
    LBFGS_ASSERT(reinterpret_cast<std::uintptr_t>(x.data()) % 64UL == 0,
                 "x must be aligned to 64-byte boundary");
    LBFGS_ASSERT(reinterpret_cast<std::uintptr_t>(y.data()) % 64UL == 0,
                 "y must be aligned to 64-byte boundary");
    auto                 n     = static_cast<int64_t>(x.size());
    auto const           a_v   = _mm256_set1_ps(a);
    auto* LBFGS_RESTRICT x_ptr = x.data();
    auto* LBFGS_RESTRICT y_ptr = y.data();
    for (; n >= 16; n -= 16, x_ptr += 16, y_ptr += 16) {
        _mm256_store_ps(
            y_ptr, _mm256_add_ps(_mm256_load_ps(y_ptr),
                                 _mm256_mul_ps(a_v, _mm256_load_ps(x_ptr))));
        _mm256_store_ps(
            y_ptr + 8,
            _mm256_add_ps(_mm256_load_ps(y_ptr + 8),
                          _mm256_mul_ps(a_v, _mm256_load_ps(x_ptr + 8))));
    }
    if (n >= 8) {
        _mm256_store_ps(
            y_ptr, _mm256_add_ps(_mm256_load_ps(y_ptr),
                                 _mm256_mul_ps(a_v, _mm256_load_ps(x_ptr))));
        n -= 8;
        x_ptr += 8;
        y_ptr += 8;
    }
    for (; n > 0; --n, ++x_ptr, ++y_ptr) {
        (*y_ptr) += a * (*x_ptr);
    }
}

LBFGS_EXPORT auto axpy(float const a, gsl::span<float const> x,
                       gsl::span<float const> y, gsl::span<float> out) noexcept
    -> void
{
    LBFGS_ASSERT(x.size() == y.size() && x.size() == out.size(),
                 "incompatible dimensions");
    LBFGS_ASSERT(
        x.size() <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
        "integer overflow");
    LBFGS_ASSERT(reinterpret_cast<std::uintptr_t>(x.data()) % 64UL == 0,
                 "x must be aligned to 64-byte boundary");
    LBFGS_ASSERT(reinterpret_cast<std::uintptr_t>(y.data()) % 64UL == 0,
                 "y must be aligned to 64-byte boundary");
    LBFGS_ASSERT(reinterpret_cast<std::uintptr_t>(out.data()) % 64UL == 0,
                 "out must be aligned to 64-byte boundary");
    auto                 n       = static_cast<int64_t>(x.size());
    auto const           a_v     = _mm256_set1_ps(a);
    auto* LBFGS_RESTRICT x_ptr   = x.data();
    auto* LBFGS_RESTRICT y_ptr   = y.data();
    auto* LBFGS_RESTRICT out_ptr = out.data();
    for (; n >= 16; n -= 16, x_ptr += 16, y_ptr += 16, out_ptr += 16) {
        _mm256_store_ps(
            out_ptr, _mm256_add_ps(_mm256_load_ps(y_ptr),
                                   _mm256_mul_ps(a_v, _mm256_load_ps(x_ptr))));
        _mm256_store_ps(
            out_ptr + 8,
            _mm256_add_ps(_mm256_load_ps(y_ptr + 8),
                          _mm256_mul_ps(a_v, _mm256_load_ps(x_ptr + 8))));
    }
    if (n >= 8) {
        _mm256_store_ps(
            out_ptr, _mm256_add_ps(_mm256_load_ps(y_ptr),
                                   _mm256_mul_ps(a_v, _mm256_load_ps(x_ptr))));
        n -= 8;
        x_ptr += 8;
        y_ptr += 8;
        out_ptr += 8;
    }
    for (; n > 0; --n, ++x_ptr, ++y_ptr) {
        (*out_ptr) = a * (*x_ptr) + (*y_ptr);
    }
}
#endif
// ================================= BLAS ================================== }}}

template <size_t Alignment>
constexpr auto align_up(size_t const value) noexcept -> size_t
{
    static_assert(Alignment != 0 && (Alignment & (Alignment - 1)) == 0,
                  "Invalid alignment");
    return (value + (Alignment - 1)) & ~(Alignment - 1);
}

} // namespace detail

// =============================== Buffers ================================= {{{
struct lbfgs_buffers_t::impl_t {
    static constexpr auto cache_line_size = 64UL;

    struct Deleter {
        template <class T> auto operator()(T* p) const noexcept
        {
            // NOLINTNEXTLINE(cppcoreguidelines-owning-memory, cppcoreguidelines-no-malloc, hicpp-no-malloc)
            std::free(p);
        }
    };

  private:
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays, hicpp-avoid-c-arrays, modernize-avoid-c-arrays)
    std::unique_ptr<float[], Deleter>     _workspace;
    std::vector<detail::iteration_data_t> _history;
    std::vector<double>                   _func_history;
    size_t                                _n;

  public:
    impl_t() noexcept : _workspace{}, _history{}, _func_history{}, _n{0} {}
    impl_t(size_t n, size_t m, size_t past) noexcept
        : _workspace{}, _history{}, _func_history{}, _n{0}
    {
        resize(n, m, past);
    }

    impl_t(const impl_t&)     = delete;
    impl_t(impl_t&&) noexcept = default;
    auto operator=(const impl_t&) -> impl_t& = delete;
    auto operator=(impl_t&&) noexcept -> impl_t& = default;
    ~impl_t() noexcept                           = default;

    auto resize(size_t const n, size_t const m, size_t const past) -> void
    {
        if (n != _n || m != _history.size()) {
            // Since _workspace may need to be re-allocated, we don't want to
            // keep dangling pointers
            _history.clear();
            _history.resize(
                m, {0.0, std::numeric_limits<double>::quiet_NaN(), {}, {}});
            _n = n;

            _workspace.reset(nullptr); // release memory before allocating more
            // We allocate enough memory so _every_ buffer is aligned to cache
            // line boundary and it's size is also a multiple of cache lines.
            // That way we can write "beyond" the buffers in 1D vector
            // algorithms without segfaulting.
            auto const size = vector_size(n) * number_vectors(m);
            _workspace      = allocate_buffer(size);
            std::memset(_workspace.get(), 0, size * sizeof(float));
            for (auto i = size_t{0}; i < _history.size(); ++i) {
                _history[i].s = get(2 * i);
                _history[i].y = get(2 * i + 1);
            }
        }
        if (past != _func_history.size()) { _func_history.resize(past); }
    }

    auto make_state() noexcept -> detail::lbfgs_state_t
    {
        constexpr auto NaN = std::numeric_limits<double>::quiet_NaN();
        auto const     m   = _history.size();
        return detail::lbfgs_state_t{
            detail::iteration_history_t{{_history}},
            {NaN, get(2 * m + 0), get(2 * m + 1)},
            {NaN, get(2 * m + 2), get(2 * m + 3)},
            get(2 * m + 4),
            detail::func_eval_history_t{{_func_history}} /**/};
    }

  private:
    static auto allocate_buffer(size_t size)
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays, hicpp-avoid-c-arrays, modernize-avoid-c-arrays)
        -> std::unique_ptr<float[], Deleter>
    {
        if (size > std::numeric_limits<size_t>::max() / sizeof(float)) {
            throw std::overflow_error{
                "integer overflow in impl_t::allocate_buffer(size_t)"};
        }
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast, cppcoreguidelines-owning-memory)
        auto* p = reinterpret_cast<float*>(
            std::aligned_alloc(cache_line_size, size * sizeof(float)));
        if (p == nullptr) { throw std::bad_alloc{}; }
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays, hicpp-avoid-c-arrays, modernize-avoid-c-arrays)
        return std::unique_ptr<float[], Deleter>{p};
    }

    static constexpr auto vector_size(size_t const n) noexcept -> size_t
    {
        return detail::align_up<cache_line_size / sizeof(float)>(n);
    }

    static constexpr auto number_vectors(size_t const m) noexcept -> size_t
    {
        return 2 * m /* s and y vectors of the last m iterations */
               + 1   /* x */
               + 1   /* x_prev */
               + 1   /* g */
               + 1   /* g_prev */
               + 1;  /* d */
    }

    auto get(size_t const i) noexcept -> gsl::span<float>
    {
        auto const size = vector_size(_n);
        LBFGS_ASSERT(i * size + _n <= size * number_vectors(_history.size()),
                     "index out of bounds");
        auto* p = _workspace.get() + i * size;
        LBFGS_ASSERT(reinterpret_cast<std::uintptr_t>(p) % cache_line_size == 0,
                     "buffer is not aligned to cache line boundary");
        LBFGS_ASSERT(std::all_of(p + _n, p + size,
                                 [](auto const x) { return x == 0.0f; }),
                     "buffer is not initialized properly");
        return {p, _n};
    }
};

auto lbfgs_buffers_t::impl() noexcept -> impl_t&
{
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    return *reinterpret_cast<impl_t*>(&_storage);
}

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init, hicpp-member-init)
LBFGS_EXPORT lbfgs_buffers_t::lbfgs_buffers_t() noexcept
{
    static_assert(sizeof(impl_t) <= sizeof(storage_type));
    static_assert(alignof(impl_t) <= alignof(storage_type));
    new (&_storage) impl_t{};
}

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init, hicpp-member-init)
LBFGS_EXPORT lbfgs_buffers_t::lbfgs_buffers_t(size_t const n, size_t const m,
                                              size_t const past)
{
    static_assert(sizeof(impl_t) <= sizeof(storage_type));
    static_assert(alignof(impl_t) <= alignof(storage_type));
    new (&_storage) impl_t{n, m, past};
}

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init, hicpp-member-init)
LBFGS_EXPORT lbfgs_buffers_t::lbfgs_buffers_t(lbfgs_buffers_t&& other) noexcept
{
    new (&_storage) impl_t{std::move(other.impl())};
}

LBFGS_EXPORT auto lbfgs_buffers_t::operator=(lbfgs_buffers_t&& other) noexcept
    -> lbfgs_buffers_t&
{
    impl() = std::move(other.impl());
    return *this;
}

LBFGS_EXPORT lbfgs_buffers_t::~lbfgs_buffers_t() noexcept { impl().~impl_t(); }

LBFGS_EXPORT auto lbfgs_buffers_t::resize(size_t const n, size_t const m,
                                          size_t const past) -> void
{
    impl().resize(n, m, past);
}

LBFGS_EXPORT auto lbfgs_buffers_t::make_state() noexcept
    -> detail::lbfgs_state_t
{
    return impl().make_state();
}

LBFGS_EXPORT auto thread_local_state(lbfgs_param_t const&   params,
                                     gsl::span<float const> x0) noexcept
    -> lbfgs_buffers_t*
{
#if defined(LBFGS_CLANG)
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wexit-time-destructors"
#endif
    static thread_local lbfgs_buffers_t buffers;
#if defined(LBFGS_CLANG)
#    pragma clang diagnostic pop
#endif
    try {
        buffers.resize(x0.size(), params.m, params.past);
    }
    catch (std::bad_alloc&) {
        return nullptr;
    }
    return std::addressof(buffers);
}
// =============================== Buffers ================================= }}}

namespace detail {
// =========================== Iteration history =========================== {{{

// NOTE: g++ complains about attributes on template arguments being ignored, but
// it is fine in this case
#if defined(LBFGS_GCC)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wignored-attributes"
#endif
LBFGS_FORCEINLINE auto emplace_back_kernel_8(
    float const* LBFGS_RESTRICT x_ptr, float const* LBFGS_RESTRICT x_prev_ptr,
    float const* LBFGS_RESTRICT g_ptr, float const* LBFGS_RESTRICT g_prev_ptr,
    float* LBFGS_RESTRICT s_ptr, float* LBFGS_RESTRICT y_ptr) noexcept
    -> std::pair<__m256d, __m256d>
{
    auto const x_f      = _mm256_load_ps(x_ptr);
    auto const x_prev_f = _mm256_load_ps(x_prev_ptr);
    auto const g_f      = _mm256_load_ps(g_ptr);
    auto const g_prev_f = _mm256_load_ps(g_prev_ptr);
    _mm256_store_ps(s_ptr, _mm256_sub_ps(x_f, x_prev_f));
    _mm256_store_ps(y_ptr, _mm256_sub_ps(g_f, g_prev_f));

    auto x1      = _mm256_cvtps_pd(_mm256_extractf128_ps(x_f, 1));
    auto x0      = _mm256_cvtps_pd(_mm256_extractf128_ps(x_f, 0));
    auto x_prev1 = _mm256_cvtps_pd(_mm256_extractf128_ps(x_prev_f, 1));
    auto x_prev0 = _mm256_cvtps_pd(_mm256_extractf128_ps(x_prev_f, 0));
    auto g1      = _mm256_cvtps_pd(_mm256_extractf128_ps(g_f, 1));
    auto g0      = _mm256_cvtps_pd(_mm256_extractf128_ps(g_f, 0));
    auto g_prev1 = _mm256_cvtps_pd(_mm256_extractf128_ps(g_prev_f, 1));
    auto g_prev0 = _mm256_cvtps_pd(_mm256_extractf128_ps(g_prev_f, 0));

    x0 = _mm256_sub_pd(x0, x_prev0);
    x1 = _mm256_sub_pd(x1, x_prev1);
    g0 = _mm256_sub_pd(g0, g_prev0);
    g1 = _mm256_sub_pd(g1, g_prev1);

    x0 = _mm256_mul_pd(x0, g0);
    x1 = _mm256_mul_pd(x1, g1);
    g0 = _mm256_mul_pd(g0, g0);
    g1 = _mm256_mul_pd(g1, g1);

    return {_mm256_add_pd(x0, x1), _mm256_add_pd(g0, g1)};
}
#if defined(LBFGS_GCC)
#    pragma GCC diagnostic pop
#endif

LBFGS_EXPORT auto iteration_history_t::emplace_back(
    gsl::span<float const> x, gsl::span<float const> x_prev,
    gsl::span<float const> g, gsl::span<float const> g_prev) noexcept -> double
{
    auto idx = back_index();
    if (_size == capacity()) { _first = sum(_first, 1); }
    else {
        ++_size;
    }

#if 1
    auto* s_ptr      = _data[idx].s.data();
    auto* y_ptr      = _data[idx].y.data();
    auto* x_ptr      = x.data();
    auto* x_prev_ptr = x_prev.data();
    auto* g_ptr      = g.data();
    auto* g_prev_ptr = g_prev.data();
    auto  s_dot_y    = _mm256_set1_pd(0.0);
    auto  y_dot_y    = _mm256_set1_pd(0.0);
    auto  n          = static_cast<int64_t>(_data[idx].s.size());
    for (; n > 0; n -= 8, s_ptr += 8, y_ptr += 8, x_ptr += 8, x_prev_ptr += 8,
                  g_ptr += 8, g_prev_ptr += 8) {
        auto const [s_dot_y_i, y_dot_y_i] = emplace_back_kernel_8(
            x_ptr, x_prev_ptr, g_ptr, g_prev_ptr, s_ptr, y_ptr);
        s_dot_y = _mm256_add_pd(s_dot_y, s_dot_y_i);
        y_dot_y = _mm256_add_pd(y_dot_y, y_dot_y_i);
    }
    auto const final_s_dot_y = hsum(s_dot_y);
    auto const final_y_dot_y = hsum(y_dot_y);

#else
    auto&      s             = _data[idx].s;
    auto&      y             = _data[idx].y;
    auto const n             = s.size();
    auto       final_s_dot_y = 0.0;
    auto       final_y_dot_y = 0.0;
    for (auto i = size_t{0}; i < n; ++i) {
        auto const s_i =
            static_cast<double>(x[i]) - static_cast<double>(x_prev[i]);
        auto const y_i =
            static_cast<double>(g[i]) - static_cast<double>(g_prev[i]);
        s[i] = static_cast<float>(s_i);
        y[i] = static_cast<float>(y_i);
        final_s_dot_y += s_i * y_i;
        final_y_dot_y += y_i * y_i;
    }
#endif

    _data[idx].s_dot_y = final_s_dot_y;
    _data[idx].alpha   = std::numeric_limits<double>::quiet_NaN();
    LBFGS_ASSERT(final_s_dot_y > 0, "something went wrong during line search");
    return final_s_dot_y / final_y_dot_y;
}

constexpr auto iteration_history_t::capacity() const noexcept -> size_type
{
    return _data.size();
}

constexpr auto iteration_history_t::size() const noexcept -> size_type
{
    return _size;
}

constexpr auto iteration_history_t::empty() const noexcept -> bool
{
    return _size == 0;
}

constexpr auto iteration_history_t::operator[](size_type const i) const noexcept
    -> iteration_data_t const&
{
    LBFGS_ASSERT(i < size(), "index out of bounds");
    return _data[i % capacity()];
}

constexpr auto iteration_history_t::operator[](size_type const i) noexcept
    -> iteration_data_t&
{
    LBFGS_ASSERT(i < size(), "index out of bounds");
    return _data[i % capacity()];
}

constexpr auto iteration_history_t::sum(size_type const a,
                                        size_type const b) const noexcept
    -> size_type
{
    auto r = a + b;
    r -= static_cast<size_type>(r >= capacity()) * capacity();
    return r;
}

constexpr auto iteration_history_t::back_index() const noexcept -> size_type
{
    return sum(_first, _size);
}

// NOTE: No we don't want to declare a destructor because it would prevent copy
// and move constructors from being constexpr
// NOLINTNEXTLINE(hicpp-special-member-functions, cppcoreguidelines-special-member-functions)
template <bool IsConst> class iteration_history_t::history_iterator {
  public:
    using type            = history_iterator<IsConst>;
    using value_type      = iteration_data_t;
    using difference_type = std::ptrdiff_t;
    using pointer = std::conditional_t<IsConst, value_type const, value_type>*;
    using reference =
        std::conditional_t<IsConst, value_type const, value_type>&;
    using iterator_category = std::bidirectional_iterator_tag;

    constexpr history_iterator() noexcept : _obj{nullptr}, _i{0} {}
    constexpr history_iterator(history_iterator const&) noexcept = default;
    constexpr history_iterator(history_iterator&&) noexcept      = default;
    constexpr auto operator  =(history_iterator const&) noexcept
        -> history_iterator& = default;
    constexpr auto operator  =(history_iterator&&) noexcept
        -> history_iterator& = default;

    constexpr auto operator*() const noexcept -> reference
    {
        LBFGS_ASSERT(_obj != nullptr && _i < _obj->size(),
                     "iterator not dereferenceable");
        return (*_obj)[_i];
    }

    constexpr auto operator-> () const noexcept -> pointer
    {
        return std::addressof(*(*this));
    }

    constexpr auto operator++() noexcept -> type&
    {
        LBFGS_ASSERT(_obj != nullptr && _i < _obj->size(),
                     "iterator not incrementable");
        ++_i;
        return *this;
    }

    constexpr auto operator++(int) noexcept -> type
    {
        auto temp{*this};
        ++(*this);
        return temp;
    }

    constexpr auto operator--() noexcept -> type&
    {
        LBFGS_ASSERT(_obj != nullptr && _i > 0, "iterator not decrementable");
        --_i;
        return *this;
    }

    constexpr auto operator--(int) noexcept -> type
    {
        auto temp{*this};
        --(*this);
        return temp;
    }

    template <bool C>
    constexpr auto operator==(history_iterator<C> const& other) const noexcept
        -> bool
    {
        LBFGS_ASSERT(_obj == other._obj, "iterators pointing to different "
                                         "containers are not comparable");
        return _i == other._i;
    }

    template <bool C>
    constexpr auto operator!=(history_iterator<C> const& other) const noexcept
        -> bool
    {
        return !(*this == other);
    }

    // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
    constexpr operator history_iterator<true>() const noexcept
    {
        return {_obj, _i};
    }

  private:
    friend iteration_history_t;
    friend class history_iterator<!IsConst>;
    using size_type = iteration_history_t::size_type;
    using container_pointer =
        std::conditional_t<IsConst, iteration_history_t const,
                           iteration_history_t>*;

    constexpr history_iterator(container_pointer obj, size_type i) noexcept
        : _obj{obj}, _i{i}
    {}

    container_pointer _obj;
    size_type         _i;
};

constexpr auto iteration_history_t::begin() const noexcept -> const_iterator
{
    return {this, _first};
}

constexpr auto iteration_history_t::begin() noexcept -> iterator
{
    return {this, _first};
}

constexpr auto iteration_history_t::end() const noexcept -> const_iterator
{
    return {this, size()};
}

constexpr auto iteration_history_t::end() noexcept -> iterator
{
    return {this, size()};
}
// =========================== Iteration history =========================== }}}

template <class Iterator>
auto apply_inverse_hessian(Iterator begin, Iterator end, double const gamma,
                           gsl::span<float> q) -> void
{
    // for i = k − 1, k − 2, . . . , k − m
    std::for_each(std::make_reverse_iterator(end),
                  std::make_reverse_iterator(begin), [q](auto& x) {
                      // alpha_i <- rho_i*s_i^T*q
                      x.alpha = detail::dot(x.s, q) / x.s_dot_y;
                      // q <- q - alpha_i*y_i
                      detail::axpy(static_cast<float>(-x.alpha), x.y, q);
                      LBFGS_TRACE("α=%f\n", x.alpha);
                  });
    // r <- H_k^0*q
    detail::scal(static_cast<float>(gamma), q);
    LBFGS_TRACE("γ=%f\n", gamma);
    //for i = k − m, k − m + 1, . . . , k − 1
    std::for_each(begin, end, [q](auto& x) {
        // beta <- rho_i * y_i^T * r
        auto const beta = detail::dot(x.y, q) / x.s_dot_y;
        // r <- r + s_i * ( alpha_i - beta)
        detail::axpy(static_cast<float>(x.alpha - beta), x.s, q);
        LBFGS_TRACE("β=%f\n", beta);
    });
    // stop with result "H_k*f_f'=q"
}

LBFGS_EXPORT auto apply_inverse_hessian(iteration_history_t&   history,
                                        double const           gamma,
                                        gsl::span<float> const q) -> void
{
    apply_inverse_hessian(history.begin(), history.end(), gamma, q);
}

} // namespace detail

LBFGS_NAMESPACE_END

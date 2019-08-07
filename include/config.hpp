// Copyright (c) 2019, Tom Westerhout
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// * Neither the name of the copyright holder nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
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

#pragma once

/// \file config.hpp
///
///

/// \cond
#if defined(LBFGS_DOXYGEN_IN_HOUSE)
#    define LBFGS_ASSUME(cond) static_cast<void>(0)
#elif defined(__clang__)
#    define LBFGS_CLANG                                                        \
        (__clang_major__ * 10000 + __clang_minor__ * 100 + __clang_patchlevel__)
#    define LBFGS_ASSUME(cond) __builtin_assume(cond)
#elif defined(__GNUC__)
#    define LBFGS_GCC                                                          \
        (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
#    define LBFGS_ASSUME(cond)                                                 \
        do {                                                                   \
            if (!(cond)) __builtin_unreachable();                              \
        } while (false)
#elif defined(_MSV_VER)
#    define LBFGS_MSVC _MSV_VER
#    define LBFGS_ASSUME(cond)                                                 \
        do {                                                                   \
        } while (false)
#else
// clang-format off
#error "Unsupported compiler. Please, submit a request to https://github.com/twesterhout/lbfgs-cpp/issues."
// clang-format on
#endif
/// \endcond

/// \cond
#if defined(LBFGS_DOXYGEN_IN_HOUSE)
#    define LBFGS_EXPORT
#    define LBFGS_NOINLINE
#    define LBFGS_FORCEINLINE inline
#    define LBFGS_LIKELY(cond) (cond)
#    define LBFGS_UNLIKELY(cond) (cond)
#    define LBFGS_CURRENT_FUNCTION ""
#elif defined(WIN32) || defined(_WIN32)
#    define LBFGS_EXPORT __declspec(dllexport)
#    define LBFGS_NOINLINE __declspec(noinline)
#    define LBFGS_FORCEINLINE __forceinline inline
#    define LBFGS_LIKELY(cond) (cond)
#    define LBFGS_UNLIKELY(cond) (cond)
#    define LBFGS_CURRENT_FUNCTION __FUNCTION__
#else
#    define LBFGS_EXPORT __attribute__((visibility("default")))
#    define LBFGS_NOINLINE __attribute__((noinline))
#    define LBFGS_FORCEINLINE __attribute__((always_inline)) inline
#    define LBFGS_LIKELY(cond) __builtin_expect(!!(cond), 1)
#    define LBFGS_UNLIKELY(cond) __builtin_expect(!!(cond), 0)
#    define LBFGS_CURRENT_FUNCTION __PRETTY_FUNCTION__
#endif
/// \endcond

/// \cond
#define LBFGS_NAMESPACE tcm::lbfgs
#define LBFGS_NAMESPACE_BEGIN                                                  \
    namespace tcm {                                                            \
    namespace lbfgs {
#define LBFGS_NAMESPACE_END                                                    \
    } /*namespace lbfgs*/                                                      \
    } /*namespace tcm*/
/// \endcond

#if defined(LBFGS_DEBUG)
/// \brief Produces some intermediate output which is useful for tracing steps
/// of the algorithm.
///
/// \note Used for debugging only
#    define LBFGS_TRACE(fmt, ...)                                              \
        do {                                                                   \
            ::std::fprintf(                                                    \
                stderr,                                                        \
                "\x1b[1m\x1b[97m%s:%i:\x1b[0m \x1b[90mtrace:\x1b[0m " fmt,     \
                __FILE__, __LINE__, __VA_ARGS__);                              \
        } while (false)
#else
#    define LBFGS_TRACE(fmt, ...) static_cast<void>(0)
#endif

/// \cond
// clang-format off
#define LBFGS_BUG_MESSAGE                                                    \
    "╔═════════════════════════════════════════════════════════════════╗\n"  \
    "║       Congratulations, you have found a bug in lbfgs-cpp!       ║\n"  \
    "║              Please, be so kind to submit it here               ║\n"  \
    "║         https://github.com/twesterhout/lbfgs-cpp/issues         ║\n"  \
    "╚═════════════════════════════════════════════════════════════════╝"
// clang-format on
/// \endcond

/// \cond
#if defined(LBFGS_CLANG)
// Clang refuses to display newlines in static_asserts
#    define LBFGS_STATIC_ASSERT_BUG_MESSAGE                                    \
        "Congratulations, you have found a bug in lbfgs-cpp! Please, be so "   \
        "kind to submit it to "                                                \
        "https://github.com/twesterhout/lbfgs-cpp/issues."
#else
#    define TCM_STATIC_ASSERT_BUG_MESSAGE "\n" TCM_BUG_MESSAGE
#endif
/// \endcond

#if !defined(LBFGS_DOXYGEN_IN_HOUSE) && defined(LBFGS_DEBUG)
#    define LBFGS_ASSERT(cond, msg)                                            \
        (LBFGS_LIKELY(cond)                                                    \
             ? static_cast<void>(0)                                            \
             : ::LBFGS_NAMESPACE::detail::assert_fail(                         \
                 #cond, __FILE__, __LINE__,                                    \
                 static_cast<char const*>(LBFGS_CURRENT_FUNCTION), msg))
#else
/// \brief A slightly nicer alternative to #assert macro from `<cassert>`.
///
/// This macro can be used in `constexpr` and `noexcept` functions.
///
/// \note Enabled only when `LBFGS_DEBUG` is defined.
#    define LBFGS_ASSERT(cond, msg) static_cast<void>(0)
#endif

LBFGS_NAMESPACE_BEGIN
/// Return codes used by LBFGS++ library
enum class status_t {
    success = 0,
    out_of_memory,
    invalid_storage_size,
    invalid_epsilon,
    invalid_delta,
    too_many_iterations,
    invalid_argument,
    rounding_errors_prevent_progress,
    maximum_step_reached,
    minimum_step_reached,
    too_many_function_evaluations,
    interval_too_small,
    invalid_interval_tolerance,
    invalid_function_tolerance,
    invalid_gradient_tolerance,
    invalid_step_bounds,
};

namespace detail {
/// \brief Terminated the program with a pretty message.
///
/// This function is called whenever an assertion fails.
[[noreturn]] auto assert_fail(char const* expr, char const* file, unsigned line,
                              char const* function, char const* msg) noexcept
    -> void;
} // namespace detail
LBFGS_NAMESPACE_END

#if defined(LBFGS_USE_GSL_LITE)

#    include <cstdlib> // std::abort

namespace gsl {
/// \brief Custom error handler for GSL contract violations.
///
/// We simply call #assert_fail().
///
/// \todo Make this optional so that projects depending on both L-BFGS++ and GSL
/// can use their own custom error handling functions.
[[noreturn]] inline constexpr auto
fail_fast_assert_handler(char const* expr, char const* msg, char const* file,
                         int const line) -> void
{
    // This is a dummy if which will always evaluate to true. We need it since
    // fail_fast_assert_handler in gsl-lite is marked constexpr and out
    // assert_fail is not.
    if (line != -2147483648) {
        ::LBFGS_NAMESPACE::detail::assert_fail(
            expr, file, static_cast<unsigned>(line), "", msg);
    }
    // This call is needed, because we mark the function [[noreturn]] and the
    // compilers don't know that line numbers can't be negative.
    std::abort();
}
} // namespace gsl

#endif // LBFGS_USE_GSL_LITE

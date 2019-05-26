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

#if defined(__clang__)
#    define LBFGSCPP_CLANG                                                     \
        (__clang_major__ * 10000 + __clang_minor__ * 100 + __clang_patchlevel__)
#    define LBFGSCPP_ASSUME(cond) __builtin_assume(cond)
#elif defined(__GNUC__)
#    define LBFGSCPP_GCC                                                       \
        (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
#    define LBFGSCPP_ASSUME(cond)                                              \
        do {                                                                   \
            if (!(cond)) __builtin_unreachable();                              \
        } while (false)
#elif defined(_MSV_VER)
#    define LBFGSCPP_MSVC _MSV_VER
#    define LBFGSCPP_ASSUME(cond)                                              \
        do {                                                                   \
        } while (false)
#else
// clang-format off
#error "Unsupported compiler. Please, submit a request to https://github.com/twesterhout/lbfgs-cpp/issues."
// clang-format on
#endif

#if defined(WIN32) || defined(_WIN32)
#    define LBFGS_EXPORT __declspec(dllexport)
#    define LBFGS_NOINLINE __declspec(noinline)
#    define LBFGS_FORCEINLINE __forceinline inline
#else
#    define LBFGS_EXPORT __attribute__((visibility("default")))
#    define LBFGS_NOINLINE __attribute__((noinline))
#    define LBFGS_FORCEINLINE __attribute__((always_inline)) inline
#endif

#define LBFGS_NAMESPACE tcm::lbfgs
#define LBFGS_NAMESPACE_BEGIN                                                  \
    namespace tcm {                                                            \
    namespace lbfgs {
#define LBFGS_NAMESPACE_END                                                    \
    } /*namespace lbfgs*/                                                      \
    } /*namespace tcm*/

#define LBFGS_TRACE(fmt, ...)                                                  \
    do {                                                                       \
        constexpr auto plain_fmt = "%s:%i trace: ";                            \
        constexpr auto pretty_fmt =                                            \
            "\x1b[1m\x1b[97m%s:%i:\x1b[0m \x1b[90mtrace:\x1b[0m ";             \
        std::fprintf(stderr, pretty_fmt, __FILE__, __LINE__);                  \
        std::fprintf(stderr, fmt, __VA_ARGS__);                                \
    } while (false)

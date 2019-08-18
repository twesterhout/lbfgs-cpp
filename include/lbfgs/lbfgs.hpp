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

#pragma once

#include "config.hpp"
#include "line_search.hpp"

#if defined(LBFGS_CLANG)
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wweak-vtables"
#    pragma clang diagnostic ignored "-Wunused-template"
#endif
#include <gsl/gsl-lite.hpp>
#if defined(LBFGS_CLANG)
#    pragma clang diagnostic pop
#endif

#include <cstring> // std::memcpy
#include <numeric>
#include <system_error>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

/// \file lbfgs.hpp
///

LBFGS_NAMESPACE_BEGIN

struct lbfgs_param_t {
    /// Number of vectors used for representing the inverse Hessian matrix.
    unsigned m;
    /// Distance for delta-based convergence tests.
    unsigned past;
    /// Maximum number of BFGS iterations to perform.
    unsigned max_iter;
    /// Parameter ε for convergence test: `‖∇f(x)‖₂ < ε·max(1, ‖x‖₂)`.
    double epsilon;
    /// Parameter δ for convergence test: |f(x - past) - f| < δ·|f|
    double delta;
    /// Search interval length threshold.
    ///
    /// The line search algorithm will stop if the search interval shrinks
    /// below this threshold.
    ///
    /// \see detail::interval_too_small_fn
    double x_tol;
    /// Parameter μ in the sufficient decrease condition
    /// (`ф(α) <= ф(0) + μ·α·ф'(0)`)
    ///
    /// \see detail::strong_wolfe_fn
    double f_tol;
    /// Parameter η in the curvature condition
    /// (`|ф'(α)| <= η·|ф'(0)|`)
    ///
    /// \see detail::strong_wolfe_fn
    double g_tol;
    /// Lower bound for the step size α.
    double step_min;
    /// Upper bound for the step size α.
    double step_max;
    /// Maximum number of function evaluations during line search.
    unsigned max_f_evals;

  private:
    explicit constexpr lbfgs_param_t(ls_param_t const& ls) noexcept
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers, readability-magic-numbers)
        : m{5}
        , past{0}
        , max_iter{0}
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers, readability-magic-numbers)
        , epsilon{1e-5}
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers, readability-magic-numbers)
        , delta{1e-5}
        , x_tol{ls.x_tol}
        , f_tol{ls.f_tol}
        , g_tol{ls.g_tol}
        , step_min{ls.step_min}
        , step_max{ls.step_max}
        , max_f_evals{ls.max_f_evals}
    {}

  public:
    /// Some sane defaults for the parameters.
    constexpr lbfgs_param_t() noexcept : lbfgs_param_t{ls_param_t{}} {}

    /// Returns the parameters for More-Thuente line search.
    [[nodiscard]] constexpr auto line_search() const noexcept -> ls_param_t
    {
        ls_param_t p;
        p.x_tol       = x_tol;
        p.f_tol       = f_tol;
        p.g_tol       = g_tol;
        p.step_min    = step_min;
        p.step_max    = step_max;
        p.max_f_evals = max_f_evals;
        return p;
    }
};

struct lbfgs_result_t {
    status_t status;   ///< Termination status
    unsigned num_iter; ///< Number of iterations
    double   func;     ///< Function value
};

struct lbfgs_buffers_t;

auto thread_local_state(lbfgs_param_t const&   params,
                        gsl::span<float const> x0) noexcept -> lbfgs_buffers_t*;

template <class Function>
auto minimize(Function value_and_gradient, lbfgs_param_t const& params,
              gsl::span<float> x) -> lbfgs_result_t;

namespace detail {
auto dot(gsl::span<float const> a, gsl::span<float const> b) noexcept -> double;
auto nrm2(gsl::span<float const> x) noexcept -> double;
auto axpy(float a, gsl::span<float const> x, gsl::span<float> y) noexcept
    -> void;
auto axpy(float a, gsl::span<float const> x, gsl::span<float const> y,
          gsl::span<float> out) noexcept -> void;
auto scal(float a, gsl::span<float> x) noexcept -> void;
auto negative_copy(gsl::span<float const> src, gsl::span<float> dst) noexcept
    -> void;

/// Checks \p p for validity.
constexpr auto check_parameters(lbfgs_param_t const& p) noexcept -> status_t
{
    if (p.m <= 0) { return status_t::invalid_storage_size; }
    if (p.epsilon < 0) { return status_t::invalid_epsilon; }
    if (p.delta < 0) { return status_t::invalid_delta; }
    if (std::isnan(p.x_tol) || p.x_tol <= 0.0) {
        return status_t::invalid_interval_tolerance;
    }
    if (std::isnan(p.f_tol) || p.f_tol <= 0.0 || p.f_tol >= 1.0) {
        return status_t::invalid_function_tolerance;
    }
    if (std::isnan(p.g_tol) || p.g_tol <= 0.0 || p.g_tol >= 1.0) {
        return status_t::invalid_gradient_tolerance;
    }
    if (std::isnan(p.step_min) || std::isnan(p.step_max) || p.step_min <= 0.0
        || p.step_max <= p.step_min) {
        return status_t::invalid_step_bounds;
    }
    return status_t::success;
}

template <class T>
LBFGS_FORCEINLINE constexpr auto as_const(T& x) noexcept -> T const&
{
    return x;
}

struct iteration_data_t {
    double           s_dot_y;
    double           alpha;
    gsl::span<float> s;
    gsl::span<float> y;
};

/// \brief A ring span of #iteration_data_t
///
///
class iteration_history_t {
    static_assert(std::is_nothrow_copy_assignable_v<iteration_data_t>);
    static_assert(std::is_nothrow_move_assignable_v<iteration_data_t>);
    static_assert(
        std::is_nothrow_copy_assignable_v<gsl::span<iteration_data_t>>);
    static_assert(
        std::is_nothrow_move_assignable_v<gsl::span<iteration_data_t>>);
    template <bool> class history_iterator;

  public:
    using value_type      = iteration_data_t;
    using reference       = iteration_data_t&;
    using const_reference = iteration_data_t const&;
    using size_type       = size_t;
    using iterator        = history_iterator<false>;
    using const_iterator  = history_iterator<true>;

    /// Constructs an empty history object.
    explicit constexpr iteration_history_t(
        gsl::span<iteration_data_t> data) noexcept
        : _first{0}, _size{0}, _data{data}
    {}

    constexpr iteration_history_t(iteration_history_t const&) noexcept =
        default;
    constexpr iteration_history_t(iteration_history_t&&) noexcept = default;
    constexpr auto operator     =(iteration_history_t const&) noexcept
        -> iteration_history_t& = default;
    constexpr auto operator     =(iteration_history_t&&) noexcept
        -> iteration_history_t& = default;

    auto emplace_back(gsl::span<float const> x, gsl::span<float const> x_prev,
                      gsl::span<float const> g,
                      gsl::span<float const> g_prev) noexcept -> double;

    [[nodiscard]] constexpr auto begin() const noexcept -> const_iterator;
    [[nodiscard]] constexpr auto begin() noexcept -> iterator;
    [[nodiscard]] constexpr auto end() const noexcept -> const_iterator;
    [[nodiscard]] constexpr auto end() noexcept -> iterator;

  private:
    [[nodiscard]] constexpr auto capacity() const noexcept -> size_type;
    [[nodiscard]] constexpr auto size() const noexcept -> size_type;
    [[nodiscard]] constexpr auto empty() const noexcept -> bool;
    [[nodiscard]] constexpr auto operator[](size_type i) const noexcept
        -> iteration_data_t const&;
    [[nodiscard]] constexpr auto operator[](size_type i) noexcept
        -> iteration_data_t&;
    [[nodiscard]] constexpr auto sum(size_type a, size_type b) const noexcept
        -> size_type;
    [[nodiscard]] constexpr auto back_index() const noexcept -> size_type;

  private:
    friend history_iterator<true>;
    friend history_iterator<false>;

    size_type                   _first;
    size_type                   _size;
    gsl::span<iteration_data_t> _data;
};

/// Keeps track of the last function evaluation results.
class func_eval_history_t {
  public:
    explicit constexpr func_eval_history_t(gsl::span<double> data) noexcept
        : _first{0}, _size{0}, _data{data}
    {}

    func_eval_history_t(func_eval_history_t const&) = delete;
    func_eval_history_t(func_eval_history_t&&)      = delete;
    auto operator=(func_eval_history_t const&) -> func_eval_history_t& = delete;
    auto operator=(func_eval_history_t &&) -> func_eval_history_t& = delete;

    [[nodiscard]] constexpr auto size() const noexcept { return _size; }
    [[nodiscard]] constexpr auto capacity() const noexcept
    {
        return _data.size();
    }

    constexpr auto emplace_back(double const func) noexcept -> void
    {
        LBFGS_ASSERT(capacity() > 0, "there is no place in the queue");
        auto const idx = sum(_first, _size);
        if (_size == capacity()) { _first = sum(_first, 1); }
        else {
            ++_size;
        }
        _data[idx] = func;
    }

    [[nodiscard]] constexpr auto front() const noexcept -> double
    {
        LBFGS_ASSERT(size() > 0, "index out of bounds");
        return _data[_first];
    }

    [[nodiscard]] constexpr auto back() const noexcept -> double
    {
        LBFGS_ASSERT(size() > 0, "index out of bounds");
        return _data[sum(_first, size() - 1)];
    }

  private:
    [[nodiscard]] constexpr auto sum(size_t const a, size_t const b) const
        noexcept -> size_t
    {
        auto r = a + b;
        r -= static_cast<size_t>(r >= capacity()) * capacity();
        return r;
    }

    size_t            _first;
    size_t            _size;
    gsl::span<double> _data;
};

/// Returns whether two spans are overlapping.
///
/// \note I'm still not 100% certain this code is guaranteed to work by the
/// standard. Here's why I think it should. Let's call endpoints of x `x₁` and
/// `x₂` and of `y` -- `y₁` and `y₂`. If `*y₁` belongs to both `x` and `y`, then
/// we can find an index `i` such that `y₁ == x₁ + i`. This implies that `x₁ <=
/// y₁` and `y₁ <= x₂`. If instead we say that somehow `x₁ <= y₁` and `y₁ <=
/// x₂`, then again we can find an index `i` such that `y₁ == x₁ + i`. But
/// according to the standard, `x₁ <= y₁` is undefined if `*x₁` and `*y₁` are
/// not part of the same array. This means that this comparison can
/// in theory return true even though `y₁` doesn't belong to `x`. But if this is
/// the case, then we now have an element in `x` which is located at the same
/// address as `*y₁`. I.e. a contradiction. Thus `y₁` does belong to `x`.
template <class T1, class T2,
          class = std::enable_if_t<
              std::is_same_v<std::remove_const_t<T1>, std::remove_const_t<T2>>>>
constexpr auto are_overlapping(gsl::span<T1> const x,
                               gsl::span<T2> const y) noexcept -> bool
{
    using T          = std::add_const_t<T1>;
    auto const x_ptr = static_cast<T*>(x.data());
    auto const y_ptr = static_cast<T*>(y.data());
    return (x_ptr <= y_ptr && y_ptr <= x_ptr + x.size())
           || (y_ptr <= x_ptr && x_ptr <= y_ptr + y.size());
}

/// \brief Point
struct lbfgs_point_t {
  private:
    double _value;          ///< Function value at #tcm::lbfgs::lbfgs_point_t::x
    gsl::span<float> _x;    ///< Point in parameter space.
    gsl::span<float> _grad; ///< Function gradient at `_x`.

    mutable double _x_norm; ///< L₂ norm of `_x`. It should be treated
        ///< as an optional value with NaN representing `nullopt`.
        ///< Prefer to use #get_x_norm() member function instead
    mutable double _grad_norm; ///< L₂ norm of `_grad`. It should be treated
        ///< as an optional value with NaN representing `nullopt`.
        ///< Prefer to use #get_x_norm() member function instead

  public:
    constexpr lbfgs_point_t(double const value, gsl::span<float> const x,
                            gsl::span<float> const grad) noexcept
        : _value{value}
        , _x{x}
        , _grad{grad}
        , _x_norm{std::numeric_limits<double>::quiet_NaN()}
        , _grad_norm{std::numeric_limits<double>::quiet_NaN()}
    {
        LBFGS_ASSERT(
            x.size() == grad.size(),
            "size of the gradient must be equal to the number of variables");
        LBFGS_ASSERT(x.size() > 0,
                     "there must be at least one variable to optimise");
    }

    lbfgs_point_t(lbfgs_point_t const& other) = delete;
    lbfgs_point_t(lbfgs_point_t&&)            = delete;

    auto operator=(lbfgs_point_t const& other) noexcept -> lbfgs_point_t&
    {
        LBFGS_ASSERT(_x.size() == other._x.size(), "incompatible sizes");
        LBFGS_ASSERT(_grad.size() == other._grad.size(), "incompatible sizes");
        LBFGS_ASSERT(!are_overlapping(_x, other._x), "overlapping ranges");
        LBFGS_ASSERT(!are_overlapping(_grad, other._grad),
                     "overlapping ranges");

        if (LBFGS_UNLIKELY(this == std::addressof(other))) { return *this; }
        _value     = other._value;
        _x_norm    = other._x_norm;
        _grad_norm = other._grad_norm;
        std::memcpy(_x.data(), other._x.data(), _x.size() * sizeof(float));
        std::memcpy(_grad.data(), other._grad.data(),
                    _grad.size() * sizeof(float));
        return *this;
    }

    constexpr auto value() const noexcept -> double { return _value; }
    constexpr auto value() noexcept -> double& { return _value; }

    constexpr auto x() const noexcept -> gsl::span<float const> { return _x; }
    constexpr auto x() noexcept -> gsl::span<float>
    {
        _x_norm = std::numeric_limits<double>::quiet_NaN();
        return _x;
    }

    constexpr auto grad() const noexcept -> gsl::span<float const>
    {
        return _grad;
    }
    constexpr auto grad() noexcept -> gsl::span<float>
    {
        _grad_norm = std::numeric_limits<double>::quiet_NaN();
        return _grad;
    }

    constexpr auto x_norm() const noexcept -> double
    {
        if (std::isnan(_x_norm)) { _x_norm = detail::nrm2(_x); }
        return _x_norm;
    }

    constexpr auto grad_norm() const noexcept -> double
    {
        if (std::isnan(_grad_norm)) { _grad_norm = detail::nrm2(_grad); }
        return _grad_norm;
    }
};

struct lbfgs_state_t {
    iteration_history_t history;
    lbfgs_point_t       current;
    lbfgs_point_t       previous;
    gsl::span<float>    direction;
    func_eval_history_t function_history;
};
} // namespace detail

struct lbfgs_buffers_t {
  private:
    struct impl_t;
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers, readability-magic-numbers)
    using storage_type = std::aligned_storage_t<64, 8>;

    storage_type _storage;

    inline auto impl() noexcept -> impl_t&;

  public:
    lbfgs_buffers_t() noexcept;
    lbfgs_buffers_t(size_t n, size_t m, size_t past);

    lbfgs_buffers_t(lbfgs_buffers_t const&) = delete;
    lbfgs_buffers_t(lbfgs_buffers_t&&) noexcept;
    auto operator=(lbfgs_buffers_t const&) -> lbfgs_buffers_t& = delete;
    auto operator=(lbfgs_buffers_t&&) noexcept -> lbfgs_buffers_t&;
    ~lbfgs_buffers_t() noexcept;

    auto resize(size_t n, size_t m, size_t past) -> void;
    auto make_state() noexcept -> detail::lbfgs_state_t;
};

#if 0
inline auto print_span(char const* prefix, gsl::span<float const> xs) -> void
{
    std::printf("%s [", prefix);
    if (!xs.empty()) {
        std::printf("%f", static_cast<double>(xs[0]));
        for (auto i = size_t{1}; i < xs.size(); ++i) {
            std::printf(", %f", static_cast<double>(xs[i]));
        }
    }
    std::printf("]\n");
}
#endif

namespace detail {
auto apply_inverse_hessian(iteration_history_t& history, double gamma,
                           gsl::span<float> q) -> void;

struct line_search_runner_fn {

    constexpr line_search_runner_fn(lbfgs_state_t&       state,
                                    lbfgs_param_t const& params) noexcept
        : _state{state}, _params{params.line_search()}, _x_tol_0{_params.x_tol}
    {}

    line_search_runner_fn(line_search_runner_fn const&) = delete;
    line_search_runner_fn(line_search_runner_fn&&)      = delete;
    auto operator                 =(line_search_runner_fn const&)
        -> line_search_runner_fn& = delete;
    auto operator=(line_search_runner_fn &&) -> line_search_runner_fn& = delete;

  private:
    template <class Function> struct wrapper_t {
        Function               value_and_gradient;
        gsl::span<float>       x;
        gsl::span<float>       grad;
        gsl::span<float const> x_0;
        gsl::span<float const> direction;

        static_assert(
            std::is_invocable_r_v<double, Function, gsl::span<float const>,
                                  gsl::span<float>>,
            "`Function` should have a signature `auto (gsl::span<float const>, "
            "gsl::span<float>) -> double`. `noexcept` qualified functions are "
            "okay. Returning a `float` which will then be converted to double "
            "is also fine.");

        static constexpr auto is_noexcept() noexcept -> bool
        {
            return std::is_nothrow_invocable_r_v<
                double, Function, gsl::span<float const>, gsl::span<float>>;
        }

        auto operator()(double const alpha,
                        std::false_type /*compute gradient*/) const
            noexcept(is_noexcept())
        {
            for (auto i = size_t{0}; i < x.size(); ++i) {
                x[i] = x_0[i] + static_cast<float>(alpha) * direction[i];
            }
            // Yes, we want implicit conversion to double here!
            double const f_x = value_and_gradient(
                static_cast<gsl::span<float const>>(x), grad);
            return f_x;
        }

        auto operator()(double const alpha,
                        std::true_type /*compute gradient*/ = {}) const
            noexcept(is_noexcept())
        {
            auto const f_x  = (*this)(alpha, std::false_type{});
            auto const df_x = detail::dot(grad, direction);
            return std::make_pair(f_x, df_x);
        }
    };

  public:
    template <class Function>
    auto operator()(Function&& value_and_gradient, double const step_0)
        -> status_t
    {
        auto const func_0 = _state.current.value();
        auto const grad_0 =
            detail::dot(_state.direction, as_const(_state.current).grad());
        LBFGS_TRACE("<line_search_runner::operator()>\n"
                    "f(x_0) = %.10e, f'(x_0) = %.10e\n",
                    func_0, grad_0);
        // constexpr auto epsilon =
        //     static_cast<double>(std::numeric_limits<float>::epsilon()) / 8.0;
        // Line search will faithfully try to optimise the function even if
        // `grad_0` is quite small, because it uses doubles everywhere. We,
        // however, know that for ‖f'(x_0)‖₂ < ε·α₀ we won't even be able to
        // notice the difference between `f(x_0)` and `f(x_0 + α₀·f'(x_0))` let
        // alone do something like cubic interpolation...
        // if (std::abs(grad_0) < epsilon * step_0) {
        //     LBFGS_TRACE("</line_search_runner::operator()>\n"
        //                 "f(x)=%.10e, f'(x)=%.10e\n",
        //                 func_0, grad_0);
        //     return status_t::rounding_errors_prevent_progress;
        // }
        _params.x_tol = _x_tol_0 * _state.current.x_norm();
        auto wrapper  = wrapper_t<Function&>{
            value_and_gradient, _state.current.x(), _state.current.grad(),
            as_const(_state.previous).x(), _state.direction};
        auto const result =
            line_search(wrapper, _params.at_zero(func_0, grad_0), step_0);
        _state.current.value() = result.func;
        if (!result.cached) {
            // Make sure that _state.current.grad is up-to-date
            wrapper(result.step, std::false_type{});
        }
        LBFGS_TRACE("</line_search_runner::operator()>\n"
                    "f(x)=%.10e, f'(x)=%.10e\n",
                    _state.current.value(), static_cast<double>(result.grad));
        return result.status;
    }

  private:
    lbfgs_state_t& _state;
    ls_param_t     _params;
    double         _x_tol_0;
};

/// \brief Checks whether we have reached a local minimum.
///
/// We perform the following check: `‖∇f‖₂ < ε·max(‖x‖₂, 1)` where
/// `ε` is #tcm::lbfgs::lbfgs_param_t::epsilon.
struct gradient_small_enough_fn {
    lbfgs_state_t const& state;
    lbfgs_param_t const& params;

    /*constexpr*/ auto operator()() const noexcept -> bool
    {
        auto const g_norm = state.current.grad_norm();
        auto const x_norm = state.current.x_norm();
        auto const result = g_norm < params.epsilon * std::max(x_norm, 1.0);
        LBFGS_TRACE("is gradient small? %.10e < %.10e * %.10e? -> %i\n", g_norm,
                    params.epsilon, std::max(x_norm, 1.0), result);
        return result;
    }
};

struct search_direction_too_small_fn {
    lbfgs_state_t const& state;
    lbfgs_param_t const& params;

    /*constexpr*/ auto operator()() const noexcept -> bool
    {
        auto const direction_norm = detail::nrm2(state.direction);
        auto const x_norm         = state.current.x_norm();
        auto const result =
            direction_norm
            < static_cast<double>(std::numeric_limits<float>::epsilon())
                  * std::max(x_norm, 1.0);
        LBFGS_TRACE("is direction small? %.10e < %.10e * %.10e? -> %i\n",
                    direction_norm,
                    static_cast<double>(std::numeric_limits<float>::epsilon()),
                    std::max(x_norm, 1.0), result);
        return result;
    }
};

struct too_little_progress_fn {
    lbfgs_state_t&       state;
    lbfgs_param_t const& params;

    constexpr auto operator()() const noexcept -> bool
    {
        auto& history = state.function_history;
        if (params.past == 0) { return false; }
        history.emplace_back(state.current.value());
        if (history.size() != history.capacity()) { return false; }
        auto const old     = history.back();
        auto const current = history.front();
        return std::abs(old - current) < params.delta * std::abs(current);
    }
};

template <class Function>
auto minimize(Function value_and_gradient, lbfgs_param_t const& params,
              lbfgs_state_t& state) -> lbfgs_result_t
{
    state.current.value() =
        value_and_gradient(as_const(state.current).x(), state.current.grad());

    gradient_small_enough_fn      gradient_is_small{state, params};
    search_direction_too_small_fn direction_is_small{state, params};
    too_little_progress_fn        too_little_progress{state, params};
    line_search_runner_fn         do_line_search{state, params};

    if (gradient_is_small()) {
        return {status_t::success, 0, state.current.value()};
    }
    auto step_0 = 1.0 / state.current.grad_norm();
    detail::negative_copy(as_const(state.current).grad(), state.direction);

    for (auto iteration = 1U;; ++iteration) {
        state.previous = state.current;

        if (auto const status = do_line_search(value_and_gradient, step_0);
            status != status_t::success) {
            // Line search "failed". If it didn't improve the loss function at
            // all, then we better undo it.
            if (std::isnan(state.current.value())
                || state.current.value() > state.previous.value()) {
                state.current = state.previous;
            }
            return {status, iteration, state.current.value()};
        }
        if (gradient_is_small() || too_little_progress()) {
            return {status_t::success, iteration, state.current.value()};
        }
        if (iteration == params.max_iter) {
            return {status_t::too_many_iterations, iteration,
                    state.current.value()};
        }

        auto const gamma = state.history.emplace_back(
            as_const(state.current).x(), as_const(state.previous).x(),
            as_const(state.current).grad(), as_const(state.previous).grad());
        detail::negative_copy(as_const(state.current).grad(), state.direction);
        apply_inverse_hessian(state.history, gamma, state.direction);
        if (direction_is_small()) {
            return {status_t::success, iteration, state.current.value()};
        }

        // From now on, always start with αₜ = 1
        step_0 = 1.0;
    }
}
} // namespace detail

template <class Function>
auto minimize(Function value_and_gradient, lbfgs_param_t const& params,
              gsl::span<float> x) -> lbfgs_result_t
{
    if (auto status = detail::check_parameters(params);
        LBFGS_UNLIKELY(status != status_t::success)) {
        return {status, 0, std::numeric_limits<double>::quiet_NaN()};
    }
    if (x.empty()) {
        // There's nothing to optimise
        return {
            status_t::success, 1,
            value_and_gradient(gsl::span<float const>{x}, gsl::span<float>{})};
    }
    auto* const buffers = thread_local_state(params, x);
    if (LBFGS_UNLIKELY(buffers == nullptr)) {
        return {status_t::out_of_memory, 0,
                std::numeric_limits<double>::quiet_NaN()};
    }
    auto state = buffers->make_state();
    std::memcpy(state.current.x().data(), x.data(), x.size() * sizeof(float));
    auto const result =
        detail::minimize(std::move(value_and_gradient), params, state);
    std::memcpy(x.data(), as_const(state.current).x().data(),
                x.size() * sizeof(float));
    return result;
}

/// #status_t can be used with `std::error_code`.
auto make_error_code(status_t) noexcept -> std::error_code;

LBFGS_NAMESPACE_END

namespace std {
/// Make `status_t` act as an error code.
template <>
struct is_error_code_enum<::LBFGS_NAMESPACE::status_t> : false_type {};
} // namespace std

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

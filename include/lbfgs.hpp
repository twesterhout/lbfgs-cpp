
#pragma once

#include "config.hpp"

// #include <algorithm>
// #include <cmath>
#include <cstring>
// #include <iterator>
#include <numeric>
// #include <optional>
#include <system_error>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <gsl/gsl-lite.hpp>
#include <cblas.h>

// #include <unistd.h>
// #include <cstdio>

#include "line_search.hpp"

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
    constexpr lbfgs_param_t(ls_param_t const& ls) noexcept
        : m{5}
        , past{0}
        , max_iter{0}
        , epsilon{1e-5}
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

    constexpr auto line_search() const noexcept -> ls_param_t
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

namespace detail {
// A hacky way of determining the integral type BLAS uses for sizes and
// increments: we pattern match on the signature of `cblas_sdot`.
template <class T> struct get_blas_int_type;

template <class T>
struct get_blas_int_type<float (*)(T, float const*, T, float const*, T)> {
    using type = T;
};

using blas_int = typename get_blas_int_type<decltype(&cblas_sdot)>::type;

auto dot(gsl::span<float const> a, gsl::span<float const> b) noexcept -> double;
auto nrm2(gsl::span<float const> x) noexcept -> double;
auto axpy(float const a, gsl::span<float const> x, gsl::span<float> y) noexcept
    -> void;
auto scal(float const a, gsl::span<float> x) noexcept -> void;
auto negative_copy(gsl::span<float const> const src,
                   gsl::span<float> const       dst) noexcept -> void;
} // namespace detail

namespace detail {
constexpr auto check_parameters(lbfgs_param_t const& p) noexcept -> status_t
{
    if (p.m <= 0) return status_t::invalid_storage_size;
    if (p.epsilon < 0) return status_t::invalid_epsilon;
    if (p.delta < 0) return status_t::invalid_delta;
    if (std::isnan(p.x_tol) || p.x_tol <= 0.0)
        return status_t::invalid_interval_tolerance;
    if (std::isnan(p.f_tol) || p.f_tol <= 0.0 || p.f_tol >= 1.0)
        return status_t::invalid_function_tolerance;
    if (std::isnan(p.g_tol) || p.g_tol <= 0.0 || p.g_tol >= 1.0)
        return status_t::invalid_gradient_tolerance;
    if (std::isnan(p.step_min) || std::isnan(p.step_max) || p.step_min <= 0.0
        || p.step_max <= p.step_min)
        return status_t::invalid_step_bounds;
    return status_t::success;
}
} // namespace detail

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
    template <bool> class history_iterator;

  public:
    using value_type      = iteration_data_t;
    using reference       = iteration_data_t&;
    using const_reference = iteration_data_t const&;
    using size_type       = size_t;
    using iterator        = history_iterator<false>;
    using const_iterator  = history_iterator<true>;

    /// Constructs an empty history object.
    constexpr iteration_history_t(gsl::span<iteration_data_t> data) noexcept
        : _first{0}, _size{0}, _data{data}
    {}

    constexpr iteration_history_t(iteration_history_t const&) noexcept =
        default;
    constexpr iteration_history_t(iteration_history_t&&) noexcept = default;
    constexpr iteration_history_t&
    operator=(iteration_history_t const&) noexcept = default;
    constexpr iteration_history_t&
    operator=(iteration_history_t&&) noexcept = default;

    auto emplace_back(gsl::span<float const> x, gsl::span<float const> x_prev,
                      gsl::span<float const> g,
                      gsl::span<float const> g_prev) noexcept -> double;

    constexpr auto begin() const noexcept -> const_iterator;
    constexpr auto begin() noexcept -> iterator;
    constexpr auto end() const noexcept -> const_iterator;
    constexpr auto end() noexcept -> iterator;

  private:
    constexpr auto emplace_back_impl(gsl::span<float const> x,
                                     gsl::span<float const> x_prev,
                                     gsl::span<float const> g,
                                     gsl::span<float const> g_prev) noexcept
        -> double;
    constexpr auto capacity() const noexcept -> size_type;
    constexpr auto size() const noexcept -> size_type;
    constexpr auto empty() const noexcept -> bool;
    constexpr auto operator[](size_type const i) const noexcept
        -> iteration_data_t const&;
    constexpr auto operator[](size_type const i) noexcept -> iteration_data_t&;
    constexpr auto sum(size_type const a, size_type const b) const noexcept
        -> size_type;
    constexpr auto back_index() const noexcept -> size_type;

  private:
    friend history_iterator<true>;
    friend history_iterator<false>;

    size_type                   _first;
    size_type                   _size;
    gsl::span<iteration_data_t> _data;
};

class func_eval_history_t {
  public:
    constexpr func_eval_history_t(gsl::span<double> data) noexcept
        : _first{0}, _size{0}, _data{data}
    {}

    func_eval_history_t(func_eval_history_t const&) = delete;
    func_eval_history_t(func_eval_history_t&&)      = delete;
    func_eval_history_t& operator=(func_eval_history_t const&) = delete;
    func_eval_history_t& operator=(func_eval_history_t&&) = delete;

    constexpr auto size() const noexcept { return _size; }
    constexpr auto capacity() const noexcept { return _data.size(); }

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

    constexpr auto front() const noexcept -> double
    {
        LBFGS_ASSERT(size() > 0, "index out of bounds");
        return _data[_first];
    }

    constexpr auto back() const noexcept -> double
    {
        LBFGS_ASSERT(size() > 0, "index out of bounds");
        return _data[sum(_first, size() - 1)];
    }

  private:
    constexpr auto sum(size_t const a, size_t const b) const noexcept -> size_t
    {
        auto r = a + b;
        r -= (r >= capacity()) * capacity();
        return r;
    }

    size_t            _first;
    size_t            _size;
    gsl::span<double> _data;
};

template <size_t Alignment>
constexpr auto align_up(size_t const value) noexcept -> size_t
{
    static_assert(Alignment != 0 && (Alignment & (Alignment - 1)) == 0,
                  "Invalid alignment");
    return (value + (Alignment - 1)) & ~(Alignment - 1);
}

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

struct lbfgs_point_t {
    double           value;
    gsl::span<float> x;
    gsl::span<float> grad;

    constexpr lbfgs_point_t(double const _value, gsl::span<float> _x,
                            gsl::span<float> _grad) noexcept
        : value{_value}, x{_x}, grad{_grad}
    {
        LBFGS_ASSERT(
            x.size() == grad.size(),
            "size of the gradient must be equal to the number of variables");
        LBFGS_ASSERT(x.size() > 0,
                     "there must be at least one variable to optimise");
    }

    lbfgs_point_t(lbfgs_point_t const& other) = delete;
    lbfgs_point_t(lbfgs_point_t&&)            = delete;

    lbfgs_point_t& operator=(lbfgs_point_t const& other) noexcept
    {
        LBFGS_ASSERT(x.size() == other.x.size(), "incompatible sizes");
        LBFGS_ASSERT(grad.size() == other.grad.size(), "incompatible sizes");
        LBFGS_ASSERT(!are_overlapping(x, other.x), "overlapping ranges");
        LBFGS_ASSERT(!are_overlapping(grad, other.grad), "overlapping ranges");

        if (LBFGS_UNLIKELY(this == std::addressof(other))) return *this;
        value = other.value;
        std::memcpy(x.data(), other.x.data(), x.size() * sizeof(float));
        std::memcpy(grad.data(), other.grad.data(),
                    grad.size() * sizeof(float));
        return *this;
    }
};

struct lbfgs_state_t {
    iteration_history_t history;
    lbfgs_point_t       current;
    lbfgs_point_t       previous;
    gsl::span<float>    direction;
    func_eval_history_t function_history;
};

struct lbfgs_buffers_t {
  private:
    std::vector<float>            _workspace;
    std::vector<iteration_data_t> _history;
    std::vector<double>           _func_history;
    size_t                        _n;

    static constexpr auto number_vectors(size_t const m) noexcept -> size_t;
    static constexpr auto vector_size(size_t const n) noexcept -> size_t;
    auto                  get(size_t const i) noexcept -> gsl::span<float>;

  public:
    lbfgs_buffers_t() noexcept;
    lbfgs_buffers_t(size_t n, size_t m, size_t past);
    auto resize(size_t n, size_t m, size_t past) -> void;
    auto make_state() noexcept -> lbfgs_state_t;
};

auto thread_local_state(lbfgs_param_t const&   params,
                        gsl::span<float const> x0) noexcept -> lbfgs_buffers_t*;

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

auto apply_inverse_hessian(iteration_history_t& history, double const gamma,
                           gsl::span<float> const q) -> void;

struct line_search_runner_fn {

    constexpr line_search_runner_fn(lbfgs_state_t&       state,
                                    lbfgs_param_t const& params) noexcept
        : _state{state}, _params{params.line_search()}
    {}

    line_search_runner_fn(line_search_runner_fn const&) = delete;
    line_search_runner_fn(line_search_runner_fn&&)      = delete;
    line_search_runner_fn& operator=(line_search_runner_fn const&) = delete;
    line_search_runner_fn& operator=(line_search_runner_fn&&) = delete;

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
        auto const func_0 = _state.current.value;
        auto const grad_0 = detail::dot(_state.direction, _state.current.grad);
        LBFGS_TRACE("f(x_0) = %f, f'(x_0) = %f\n", func_0, grad_0);
        print_span("grad = ", _state.current.grad);
        auto wrapper = wrapper_t<Function&>{
            value_and_gradient, _state.current.x, _state.current.grad,
            _state.previous.x, _state.direction};
        auto const result =
            line_search(wrapper, _params.at_zero(func_0, grad_0), step_0);
        _state.current.value = result.func;
        if (!result.cached) { wrapper(result.step, std::false_type{}); }
        LBFGS_TRACE("f(x)=%f, f'(x)=%f\n", _state.current.value,
                    static_cast<double>(result.grad));
        print_span("x = ", _state.current.x);
        return result.status;
    }

  private:
    lbfgs_state_t& _state;
    ls_param_t     _params;
};

struct gradient_small_enough_fn {
    lbfgs_state_t const& state;
    lbfgs_param_t const& params;

    /*constexpr*/ auto operator()(double const g_norm) const noexcept -> bool
    {
        auto const x_norm = detail::nrm2(state.current.x);
        return g_norm < params.epsilon * std::max(x_norm, 1.0);
    }

    /*constexpr*/ auto operator()() const noexcept -> bool
    {
        auto const g_norm = detail::nrm2(state.current.grad);
        return (*this)(g_norm);
    }
};

struct too_little_progress_fn {
    lbfgs_state_t&       state;
    lbfgs_param_t const& params;

    constexpr auto operator()() const noexcept -> bool
    {
        auto& history = state.function_history;
        if (params.past == 0) { return false; }
        history.emplace_back(state.current.value);
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
    state.current.value = value_and_gradient(
        gsl::span<float const>{state.current.x}, state.current.grad);

    gradient_small_enough_fn gradient_is_small{state, params};
    too_little_progress_fn   too_little_progress{state, params};
    line_search_runner_fn    do_line_search{state, params};

    auto const grad_0_norm = detail::nrm2(state.current.grad);
    if (gradient_is_small(grad_0_norm)) {
        return {status_t::success, 0, state.current.value};
    }
    auto step = 1.0 / grad_0_norm;
    detail::negative_copy(state.current.grad, state.direction);

    for (auto iteration = 1u;; ++iteration) {
        state.previous = state.current;

        if (auto const status = do_line_search(value_and_gradient, step);
            status != status_t::success) {
            return {status, iteration, state.current.value};
        }
        if (gradient_is_small()) {
            return {status_t::success, iteration, state.current.value};
        }
        if (too_little_progress()) {
            return {status_t::success, iteration, state.current.value};
        }
        if (iteration == params.max_iter) {
            return {status_t::too_many_iterations, iteration,
                    state.current.value};
        }

        auto const gamma =
            state.history.emplace_back(state.current.x, state.previous.x,
                                       state.current.grad, state.previous.grad);
        detail::negative_copy(state.current.grad, state.direction);
        apply_inverse_hessian(state.history, gamma, state.direction);

        // From now on, always start with αₜ = 1
        step = 1.0;
    }
}

template <class Function>
auto minimize(Function value_and_gradient, lbfgs_param_t const& params,
              gsl::span<float> x) -> lbfgs_result_t
{
    if (auto status = detail::check_parameters(params);
        LBFGS_UNLIKELY(status != status_t::success)) {
        return {status, 0, std::numeric_limits<double>::quiet_NaN()};
    }
    auto* const buffers = thread_local_state(params, x);
    if (LBFGS_UNLIKELY(buffers == nullptr)) {
        return {status_t::out_of_memory, 0,
                std::numeric_limits<double>::quiet_NaN()};
    }
    auto state = buffers->make_state();
    std::memcpy(state.current.x.data(), x.data(), x.size() * sizeof(float));
    auto const result = minimize(std::move(value_and_gradient), params, state);
    std::memcpy(x.data(), state.current.x.data(), x.size() * sizeof(float));
    return result;
}

LBFGS_NAMESPACE_END

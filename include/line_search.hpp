
#pragma once

#include "config.hpp"

#include <algorithm>
#include <cmath>
#include <optional>
#include <system_error>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <unistd.h>
#include <cstdio>

#include <iostream>

/// \file line_search.hpp
///
///

LBFGS_NAMESPACE_BEGIN

// ========================== Public Interface ============================= {{{

enum class status_t {
    success = 0,
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

/// Parameters for the More-Thuente line search algorithm.
struct ls_param_t {
    /// \brief Search interval length threshold.
    ///
    /// The line search algorithm will stop if the search interval shrinks
    /// below this threshold. Setting #x_tol to 0 will disable the check.
    ///
    /// \pre `x_tol >= 0`
    /// \see detail::interval_too_small_fn
    double x_tol;
    /// \brief Parameter μ in the sufficient decrease condition
    /// (`ф(α) <= ф(0) + μ·α·ф'(0)`)
    ///
    /// \pre `0 < f_tol && f_tol < 1`
    /// \see detail::strong_wolfe_fn
    double f_tol;
    /// \brief Parameter η in the curvature condition
    /// (`|ф'(α)| <= η·|ф'(0)|`)
    ///
    /// \pre `0 < g_tol && g_tol < 1`
    /// \see detail::strong_wolfe_fn
    double g_tol;
    /// \brief Lower bound for the step size α.
    ///
    /// \pre `0 < step_min && step_min < step_max`
    /// \see detail::reached_min_step_fn
    double step_min;
    /// \brief Upper bound for the step size α.
    ///
    /// \pre `0 < step_min && step_min < step_max`
    /// \see detail::reached_min_step_fn
    double step_max;
    /// Maximum number of function evaluations.
    unsigned max_f_evals;
    /// Function value at `α = 0` (i.e. `ф(0)`)
    double func_0;
    /// Derivative at `α = 0` (i.e. `ф(0)`)
    double grad_0;

    /// \brief Some sane defaults for More-Thuente line search algorithm.
    ///
    /// Pong
    constexpr ls_param_t() noexcept
        : x_tol{1e-8}
        , f_tol{1e-4}
        , g_tol{0.9}
        , step_min{1e-8}
        , step_max{1e8}
        , max_f_evals{20}
        , func_0{std::numeric_limits<double>::quiet_NaN()}
        , grad_0{std::numeric_limits<double>::quiet_NaN()}
    {}

    /// A convenience function for setting the initial state.
    ///
    /// A common use case is to pass `ls_param_t{}.at_zero(my_value,
    /// my_grad)` to the line search algorithm.
    constexpr auto at_zero(double const func, double const grad) noexcept
        -> ls_param_t&
    {
        func_0 = func;
        grad_0 = grad;
        return *this;
    }
};

struct ls_result_t {
    status_t status;      ///< Termination status
    double   step;        ///< Final step size: `α`
    double   func;        ///< Function value at #step: `ф(α)`
    double   grad;        ///< Derivative at #step: `ф'(α)`
    unsigned num_f_evals; ///< Number of function evaluations during
                          ///< the optimisation procedure.
    /// Line search is typically used as part of other non-linear optimisation
    /// algorithms. There gradient is a (possibly) high-dimensional vector. The
    /// user may want to cache the last computed gradient for further use.
    /// #cached attribute indicated whether the last function evaluation
    /// was at #step. In other words, `cached == true` means the user can
    /// safely reuse the last computed gradient.
    bool cached;
};

template <class Function>
LBFGS_NOINLINE auto
line_search(Function&& value_and_gradient, ls_param_t const& params,
            double const alpha_0 =
                1.0) noexcept(noexcept(std::
                                           declval<Function&>()(
                                               std::declval<double>())))
    -> ls_result_t;

// ========================== Public Interface ============================= }}}

/// #status_t can be used with `std::error_code`.
auto make_error_code(status_t) noexcept -> std::error_code;

LBFGS_NAMESPACE_END

namespace std {
template <>
struct is_error_code_enum<::LBFGS_NAMESPACE::status_t> : false_type {};
} // namespace std

LBFGS_NAMESPACE_BEGIN

namespace detail {

// ========================= Some data structures ========================== {{{

/// \brief Line search interval `I`.
///
/// #interval_t maintains the property that `min() <= max()`.
class interval_t {
  private:
    double _min; ///< Lower bound
    double _max; ///< Upper bound

  private:
    constexpr interval_t(double const min, double const max) noexcept
        : _min{min}, _max{max}
    {
        LBFGS_ASSERT(_min <= _max, "invalid interval");
    }

  public:
    /// \brief Creates an empty interval
    constexpr interval_t() noexcept : interval_t{0.0, 0.0} {}

    /// \brief Updates the search interval using the current state
    ///
    /// \param x is \f$\alpha_x\f$
    /// \param y is \f$\alpha_y\f$
    /// \param t is \f$\alpha_t\f$
    /// \param bracketed whether the search interval has been bracketed (i.e. we
    /// know that `t` belongs to the search interval).
    constexpr interval_t(double const x, double const y, double const t,
                         bool const bracketed) noexcept
        : interval_t{}
    {
        if (bracketed) {
            LBFGS_ASSERT(std::min(x, y) <= t && t <= std::max(x, y),
                         "invalid αₜ");
            std::tie(_min, _max) = std::minmax(x, y);
        }
        else {
            // See the explanation around eq. (2.2) on p. 291
            _min = x;
            _max = t + 4.0 * (t - x);
            LBFGS_ASSERT(_min <= _max, "invalid interval");
        }
    }

    constexpr interval_t(interval_t const&) noexcept = default;
    constexpr interval_t(interval_t&&) noexcept      = default;
    constexpr interval_t& operator=(interval_t const&) noexcept = default;
    constexpr interval_t& operator=(interval_t&&) noexcept = default;

    constexpr auto length() const noexcept { return _max - _min; }
    constexpr auto min() const noexcept { return _min; }
    constexpr auto max() const noexcept { return _max; }
};

/// \brief A point of the search interval.
///
/// It is used for two endpoints of `I` and `αₜ`.
struct endpoint_t {
    double alpha; ///< α
    double func;  ///< either ф(α) or ψ(α) depending on the stage
    double grad;  ///< either ф'(α) or ψ'(α) depending on the stage
};

/// \brief Internal state of the line search algorithm.
struct ls_state_t {
    endpoint_t x;           ///< Endpoint with the least function value
    endpoint_t y;           ///< The other endpoint
    endpoint_t t;           ///< Trial value
    interval_t interval;    ///< Interval for line search
    bool       bracketed;   ///< Whether the trial value is bracketed
    unsigned   num_f_evals; ///< Current number of function evaluations
};

/// \brief Checks the user-provided parameters for sanity.
///
/// If no problems with the input parameters can be found, #status_t::success is
/// returned. Otherwise, the #status_t returned indicated the error.
constexpr auto check_parameters(ls_param_t const& p) noexcept -> status_t
{
    if (std::isnan(p.x_tol) || p.x_tol <= 0.0)
        return status_t::invalid_interval_tolerance;
    if (std::isnan(p.f_tol) || p.f_tol <= 0.0 || p.f_tol >= 1.0)
        return status_t::invalid_function_tolerance;
    if (std::isnan(p.g_tol) || p.g_tol <= 0.0 || p.g_tol >= 1.0)
        return status_t::invalid_gradient_tolerance;
    if (std::isnan(p.step_min) || std::isnan(p.step_max) || p.step_min <= 0.0
        || p.step_max <= p.step_min)
        return status_t::invalid_step_bounds;
    if (std::isnan(p.func_0) || std::isnan(p.grad_0) || p.grad_0 >= 0.0)
        return status_t::invalid_argument;
    return status_t::success;
}

// ========================= Some data structures ========================== }}}

auto update_trial_value_and_interval(ls_state_t& state) noexcept -> void;

/// Using Maple
///
/// ```
/// > p := A * x^2 + B * x + C;
/// > solution := solve([eval(p, x=a) = f_a,
///                      eval(p, x=b) = f_b,
///                      eval(diff(p, x), x=a) = df_a], [A, B, C]);
/// > A_best, B_best, C_best := op(map(t -> eval(x, x=rhs(t)), op(solution)));
/// > x_min := simplify(-B_best / (2 * A_best));
///
///                  2    2
///                (a  - b ) df_a - 2 a (f_a - f_b)
///       x_min := --------------------------------
///                (2 a - 2 b) df_a - 2 f_a + 2 f_b
///
/// > A_best;
///
///       a df_a - b df_a - f_a + f_b
///       ---------------------------
///              2            2
///             a  - 2 a b + b
/// ```
///
/// Simplifying the expression for `x_min` a bit further, we get
///
/// ```
///                          (b - a) df_a
///       x_min := a + 0.5 ----------------
///                               f_b - f_a
///                        df_a - ---------
///                                 b - a
/// ```
///
/// For `x_min` to be the minimum of `p`, we need `A_best > 0`, i.e.
/// ```
///       f_b - f_a
///       --------- > df_a
///         b - a
/// ```
///
/// We are given two endpoints `a` and `b` of an interval, values of `f` at
/// `a` and `b` and the value of `df/dx` at `a`. We then compute quadratic
/// interpolation of `f` and return the coordinate `α` of its minimum.
///
/// \precondition `f_a < f_b` which follows from Case 1 on p.299.
/// \precondition `(b - a) * df_a` which is necessary for the parabola to
///               have a minimum in the interval `[min(a, b), max(a, b)]`.
constexpr auto
minimise_quadratic_interpolation(double const a, double const f_a,
                                 double const df_a, double const b,
                                 double const f_b) noexcept -> double
{
    LBFGS_ASSERT(f_a < f_b, "Precondition violated");
    LBFGS_ASSERT((b - a) * df_a < 0.0, "Precondition violated");
    auto const length = b - a;
    auto const scale  = (f_b - f_a) / length / df_a;
    auto const alpha  = a + 0.5 * length / (1.0 - scale);
    LBFGS_ASSERT(std::min(a, b) < alpha && alpha < std::max(a, b),
                 "Postcondition violated");
    return alpha;
}

/// Again, using Maple
///
/// ```
/// > p := A * x^2 + B * x + C;
/// > solution := solve([eval(diff(p, x), x=a) = df_a,
///                      eval(diff(p, x), x=b) = df_b], [A, B, C]);
/// > A_best, B_best, C_best := op(map(t -> eval(x, x=rhs(t)), op(solution)));
/// > x_min := simplify(-B_best / (2 * A_best));
///
///                -a df_b + b df_a
///       x_min := ----------------
///                  df_a - df_b
///
/// > A_best;
///
///            df_a - df_b
///            -----------
///             2 (a - b)
///
/// ```
///
/// Simplifying the expression for `x_min` a bit further, we get
///
/// ```
///                               1
///       x_min := a + (b - a) --------
///                                df_b
///                            1 - ----
///                                df_a
/// ```
///
/// We are given two endpoints `a` and `b` of an interval and values of
/// `df/dx` at `a` and `b`. We then compute quadratic interpolation of `f`
/// and return the coordinate of its minimum.
///
/// \precondition `(df_b - df_a) * (b - a) > 0` which is necessary for the
///               parabola to be bounded from below.
constexpr auto
minimise_quadratic_interpolation(double const a, double const df_a,
                                 double const b, double const df_b) noexcept
    -> double
{
    LBFGS_ASSERT((df_b - df_a) * (b - a) > 0.0, "Precondition violated");
    auto const length = b - a;
    return a + length / (1.0 - df_b / df_a);
}

/// From https://en.wikipedia.org/wiki/Cubic_Hermite_spline we know that Cubic
/// interpolation of ``f`` is
///
///     P(x) = h₀₀(t)·f(a) + h₁₀(t)·(b - a)·f'(a)
///                + h₀₁(t)·f(b) + h₁₁(t)·(b - a)·f'(b) ,
///          where  t = (x - a) / (b - a),
///                 h₀₀(t) = 2t³ - 3t² + 1,
///                 h₁₀(t) = t³ - 2t² + t,
///                 h₀₁(t) = -2t³ + 3t²,
///                 h₁₁(t) = t³ - t².
///
/// We then solve ``dP(x)/dx = 0`` for ``x`` to obtain
///
///     x = a + (p/q)·(b - a)
///
///         where   p = θ + γ - f'(a),
///                 q = 2·γ + f'(b) - f'(a),
///                 γ = ± sqrt(θ² - f'(a)·f'(b)),
///                 θ = 3·(f(a) - f(b)) / (b - a) + f'(a) + f'(b).
///
/// We can check it using Maple
/// ```
/// > h00 := 2 * t^3 - 3 * t^2 + 1;
/// > h10 := t^3 - 2 * t^2 + t;
/// > h01 := -2 * t^3 + 3 * t^2;
/// > h11 := t^3 - t^2;
/// > t := (x - a) / (b - a);
/// > P_ := h00 * f_a + h10 * (b - a) * df_a + h01 * f_b + h11 * (b - a) * df_b;
/// > simplify([eval(P_, x=a), eval(P_, x=b),
///             eval(diff(P_, x), x=a), eval(diff(P_, x), x=b)]);
///
///             [f_a, f_b, df_a, df_b]
///
/// > theta := 3 * (f_a - f_b) / (b - a) + df_a + df_b;
/// > local gamma := sqrt(theta^2 - df_a * df_b);
/// > p := theta + gamma - df_a;
/// > q := 2 * gamma + df_b - df_a;
/// > simplify(eval(diff(P_, x), x=a + (p/q)*(b - a)));
///
///             0
/// ```
/// If we then compute the second derivative ``d²P(x)/dx²``, we'll see that
/// only ``γ * (a - b) > 0`` results in positive curvatuve.
///
inline auto minimise_cubic_interpolation(double const a, double const f_a,
                                         double const df_a, double const b,
                                         double const f_b,
                                         double const df_b) noexcept -> double
{
    LBFGS_ASSERT(a != b, "Precondition violated");
    auto const length = b - a;
    auto const temp   = 3.0 * (f_a - f_b) / length + df_b;
    auto const theta  = temp + df_a;
    auto       gamma  = theta * theta - df_a * df_b;
    if (gamma >= 0.0) {
        gamma = std::sqrt(gamma);
        if (b < a) { gamma = -gamma; }
        auto const p = temp + gamma;
        auto const q = 2.0 * gamma + df_b - df_a;
        LBFGS_ASSERT(q != 0.0, "division by zero");
        return a + p / q * length;
    }
    constexpr auto inf = std::numeric_limits<double>::infinity();
    return df_a > 0.0 ? -inf : inf;
}

struct interval_too_small_fn {
    ls_result_t&      result; ///< Reference to the return type
    ls_state_t const& state;  ///< Current state
    ls_param_t const& params; ///< Algorithm options

    /// \brief Checks whether the search interval has shrunk below the threshold
    /// which is given by #params.x_tol.
    ///
    /// If the interval is indeed too small, #result is set to the best state
    /// obtained so far and a pointer to it is returned.
    constexpr auto operator()() const noexcept -> ls_result_t*
    {
        if (state.bracketed
            && (state.interval.length()
                <= params.x_tol * state.interval.max())) {
            LBFGS_TRACE("interval too small: %f <= %f", state.interval.length(),
                        params.x_tol * state.interval.max());
            // NOTE: We return αₓ rather than αₜ!
            result = {status_t::interval_too_small,
                      state.x.alpha,
                      state.x.func,
                      state.x.grad,
                      state.num_f_evals,
                      false};
            return &result;
        }
        return nullptr;
    }
};

struct too_many_f_evals_fn {
    ls_result_t&      result; ///< Reference to the return type
    ls_state_t const& state;  ///< Current state
    ls_param_t const& params; ///< Algorithm options

    /// \brief Checks whether all available function evaluations (i.e.
    /// #params.max_f_evals) have been exhausted.
    ///
    /// If we have already evaluated the user-defined `f` #params.max_f_evals
    /// times, #result is set to the best state obtained so far and a pointer to
    /// it is returned.
    constexpr auto operator()() const noexcept -> ls_result_t*
    {
        LBFGS_ASSERT(state.num_f_evals <= params.max_f_evals, "invalid state");
        if (state.num_f_evals == params.max_f_evals) {
            LBFGS_TRACE("too many function evaluations: %u == %u\n",
                        state.num_f_evals, params.max_f_evals);
            // NOTE: We return αₓ rather than αₜ!
            result = {status_t::too_many_function_evaluations,
                      state.x.alpha,
                      state.x.func,
                      state.x.grad,
                      state.num_f_evals,
                      false};
            return &result;
        }
        return nullptr;
    }
};

struct reached_max_step_fn {
    ls_result_t&      result; ///< Reference to the return type
    ls_state_t const& state;  ///< Current state
    ls_param_t const& params; ///< Algorithm options

    /// \brief Checks whether the algorithm should terminate at `αₘₐₓ`.
    ///
    /// This check corresponds to the first case in Theorem 2.2 (p. 292):
    /// we reached `αₘₐₓ` and condition (2.3) holds (i.e. `ψ(αₜ) <= 0`
    /// and `ψ'(αₜ) < 0`). If this is indeed the case, #result is set to
    /// αₜ and a pointer to it is returned.
    constexpr auto operator()() const noexcept -> ls_result_t*
    {
        auto const grad_test = params.f_tol * params.grad_0;
        auto const func_test = params.func_0 + state.t.alpha * grad_test;
        if (state.t.alpha == params.step_max && state.t.func <= func_test
            && state.t.grad <= grad_test) {
            LBFGS_TRACE("reached αₘₐₓ: ф'(αₜ)=%f <= %f\n", state.t.grad,
                        grad_test);
            result = {status_t::maximum_step_reached,
                      state.t.alpha,
                      state.t.func,
                      state.t.grad,
                      state.num_f_evals,
                      false};
            return &result;
        }
        return nullptr;
    }
};

struct reached_min_step_fn {
    ls_result_t&      result; ///< Reference to the return type
    ls_state_t const& state;  ///< Current state
    ls_param_t const& params; ///< Algorithm options

    /// \brief Checks whether the algorithm should terminate at `αₘᵢₙ`.
    ///
    /// This check corresponds to the second case in Theorem 2.2 (p. 292): we
    /// reached `αₘᵢₙ` and condition (2.4) holds (i.e. `ψ(αₜ) > 0` and
    /// `ψ'(αₜ) >= 0`). If this is indeed the case, #result is set to αₜ and a
    /// pointer to it is returned.
    constexpr auto operator()() const noexcept -> ls_result_t*
    {
        auto const grad_test = params.f_tol * params.grad_0;
        auto const func_test = params.func_0 + state.t.alpha * grad_test;
        if (state.t.alpha == params.step_min
            && (state.t.func > func_test || state.t.grad >= grad_test)) {
            LBFGS_TRACE("reached αₘᵢₙ: ф'(αₜ)=%f >= %f\n", state.t.grad,
                        grad_test);
            result = {status_t::minimum_step_reached,
                      state.t.alpha,
                      state.t.func,
                      state.t.grad,
                      state.num_f_evals,
                      false};
            return &result;
        }
        return nullptr;
    }
};

struct strong_wolfe_fn {
    ls_result_t&      result; ///< Reference to the return type
    ls_state_t const& state;  ///< Current state
    ls_param_t const& params; ///< Algorithm options

    /*constexpr*/ auto operator()() const noexcept -> ls_result_t*
    {
        // Both sufficient decrease (1.1) and curvatuve (1.2) conditions
        // hold (i.e. `ψ(αₜ) <= 0` and `|ф'(αₜ)| <= η·|ф'(0)|`). We use the
        // fact that `|ф'(0)| = -ф'(0)`, because `ф'(0) < 0` is a
        // precondition for the whole algorithm.
        auto const grad_test = params.f_tol * params.grad_0;
        auto const func_test = params.func_0 + state.t.alpha * grad_test;
        LBFGS_TRACE("strong_wolfe_fn: f(x)=%f, func_test=%f, |grad|=%f, "
                    "g_tol=%f, -grad_0=%f\n",
                    state.t.func, func_test, std::abs(state.t.grad),
                    params.g_tol, -params.grad_0);
        if ((state.t.func <= func_test)
            && (std::abs(state.t.grad) <= params.g_tol * (-params.grad_0))) {
            LBFGS_TRACE("Strong Wolfe conditions satisfied:\n"
                        "    sufficient decrease: %f <= %f\n"
                        "    curvature condition: %f <= %f\n"
                        "    %u function evaluations\n",
                        state.t.func, func_test, std::abs(state.t.grad),
                        params.g_tol * (-params.grad_0), state.num_f_evals);
            result = {status_t::success, state.t.alpha,     state.t.func,
                      state.t.grad,      state.num_f_evals, false};
            return &result;
        }
        return nullptr;
    }
};

struct rounding_errors_fn {
    ls_result_t&      result; ///< Reference to the return type
    ls_state_t const& state;  ///< Current state
    ls_param_t const& params; ///< Algorithm options

    constexpr auto operator()() const noexcept -> ls_result_t*
    {
        if (state.bracketed && state.t.alpha == state.interval.min()) {
            result = {status_t::rounding_errors_prevent_progress,
                      state.x.alpha,
                      state.x.func,
                      state.x.grad,
                      state.num_f_evals,
                      false};
            return &result;
        }
        return nullptr;
    }
};

struct ensure_shrinking_fn {
    /// \brief A two-element FIFO queue.
    ///
    /// The purpose of #width_history_t is to keep track of the evolution of the
    /// width of the search interval. Once the interval has been bracketed, we want
    /// it's width to keep decreasing exponentially so that the search algorithm
    /// terminates in a finite number of steps.
    ///
    /// Whenever the width doesn't decrease sufficiently fast for two "iterations"
    /// of the main loop, we use a bisection step to keep the interval shrinking.
    struct width_history_t {
        double previous; ///< Previous width
        double current;  ///< Current width

        /// Adds an element to the queue. The #previous width is forgotten.
        constexpr auto push(double const new_width) noexcept -> void
        {
            LBFGS_ASSERT(new_width >= 0, "width cannot be negative");
            LBFGS_ASSERT(
                new_width <= current,
                "width of the search interval should be non-increasing");
            previous = current;
            current  = new_width;
        }
    };

    constexpr ensure_shrinking_fn(ls_state_t&  state,
                                  double const max_width) noexcept
        : _state{state}, _width_history{2.0 * max_width, max_width}
    {
        LBFGS_ASSERT(max_width >= 0.0, "width cannot be negative");
    }

    constexpr ensure_shrinking_fn(ls_state_t&       state,
                                  ls_param_t const& params) noexcept
        : ensure_shrinking_fn{state, params.step_max - params.step_min}
    {}

    constexpr auto operator()() noexcept -> void
    {
        if (_state.bracketed) {
            // Last paragraph of p.292:
            // <quote>If the algorithm generates an interval I in [αₘᵢₙ,
            // αₘₐₓ], then we need a third rule to guarantee that the choice
            // of αₜ forces the length of I to zero. In our implementation
            // this is done by monitoring the length of I; if the length of
            // I does not decrease by a factor δ < 1 after two trials, then
            // a bisection step is used for the next trial αₜ.</quote>
            constexpr auto delta = 0.66;
            auto const new_width = std::abs(_state.y.alpha - _state.x.alpha);
            if (new_width >= delta * _width_history.previous) {
                _state.t.alpha =
                    _state.x.alpha + 0.5 * (_state.y.alpha - _state.x.alpha);
            }
            _width_history.push(new_width);
        }
    }

  private:
    ls_state_t&     _state;
    width_history_t _width_history;
};

struct with_modified_function_t {
    constexpr with_modified_function_t(ls_state_t&  s,
                                       double const grad_test) noexcept
        : _s{s}, _grad_test{grad_test}
    {
        _s.x.func -= _s.x.alpha * _grad_test;
        _s.x.grad -= _grad_test;
        _s.y.func -= _s.y.alpha * _grad_test;
        _s.y.grad -= _grad_test;
        _s.t.func -= _s.t.alpha * _grad_test;
        _s.t.grad -= _grad_test;
    }

    ~with_modified_function_t() noexcept
    {
        _s.x.func += _s.x.alpha * _grad_test;
        _s.x.grad += _grad_test;
        _s.y.func += _s.y.alpha * _grad_test;
        _s.y.grad += _grad_test;
        LBFGS_ASSERT(
            std::isnan(_s.t.func),
            "function value at αₜ is not known and should be set to NaN");
        LBFGS_ASSERT(std::isnan(_s.t.grad),
                     "derivative at αₜ is not known and should be set to NaN");
    }

    with_modified_function_t(with_modified_function_t const&) = delete;
    with_modified_function_t(with_modified_function_t&&)      = delete;
    with_modified_function_t&
                              operator=(with_modified_function_t const&) = delete;
    with_modified_function_t& operator=(with_modified_function_t&&) = delete;

  private:
    ls_state_t&  _s;
    double const _grad_test;
};

template <class Function> struct evaluate_fn {
    Function&   value_and_gradient;
    ls_state_t& state;

    constexpr auto operator()() const noexcept -> void
    {
        LBFGS_ASSERT(!std::isnan(state.t.alpha) && std::isnan(state.t.func)
                         && std::isnan(state.t.grad),
                     "invalid state");
        LBFGS_ASSERT(!state.bracketed
                         || (state.interval.min() < state.t.alpha
                             && state.t.alpha < state.interval.max()),
                     "αₜ ∉ I");
        std::tie(state.t.func, state.t.grad) =
            value_and_gradient(state.t.alpha);
        ++state.num_f_evals;
        LBFGS_TRACE("another function evaluation: αₜ=%f, ф(αₜ)=%f, ф'(αₜ)=%f\n",
                    state.t.alpha, state.t.func, state.t.grad);
    }
};

template <class Function>
evaluate_fn(Function&, ls_state_t&)->evaluate_fn<Function>;

template <class Function>
auto line_search(
    Function value_and_gradient, ls_param_t const& params,
    double const alpha_0) noexcept(noexcept(std::
                                                declval<Function&>()(
                                                    std::declval<double>())))
    -> ls_result_t
{
    constexpr auto NaN = std::numeric_limits<double>::quiet_NaN();
    LBFGS_ASSERT(check_parameters(params) == status_t::success,
                 "invalid parametes");
    LBFGS_ASSERT(alpha_0 > 0, "invalid initial step");
    auto state = ls_state_t{
        /*x=*/endpoint_t{0, params.func_0, params.grad_0},
        /*y=*/endpoint_t{0, params.func_0, params.grad_0},
        /*t=*/endpoint_t{alpha_0, NaN, NaN},
        /*interval=*/interval_t{0, 0, alpha_0, false},
        /*bracketed=*/false,
        /*num_f_evals=*/0u,
    };
    auto first_stage = true; // TODO(twesterhout): Explain stages

    ls_result_t           _result;
    evaluate_fn           evaluate{value_and_gradient, state};
    ensure_shrinking_fn   ensure_shrinking{state, params};
    interval_too_small_fn interval_too_small{_result, state, params};
    too_many_f_evals_fn   too_many_f_evals{_result, state, params};
    reached_max_step_fn   reached_max_step{_result, state, params};
    reached_min_step_fn   reached_min_step{_result, state, params};
    rounding_errors_fn    rounding_errors{_result, state, params};
    strong_wolfe_fn       strong_wolfe{_result, state, params};

    for (;;) {
        state.t.alpha =
            std::clamp(state.t.alpha, params.step_min, params.step_max);
        LBFGS_TRACE("proposed αₜ=%f, I = [%f, %f]\n", state.t.alpha,
                    state.interval.min(), state.interval.max());
        if (auto r = interval_too_small(); r) { return *r; }
        if (auto r = too_many_f_evals(); r) { return *r; }
        if (auto r = rounding_errors(); r) { return *r; }
        evaluate();
        if (auto r = reached_max_step(); r) { return *r; }
        if (auto r = reached_min_step(); r) { return *r; }
        if (auto r = strong_wolfe(); r) { return *r; }
        // In the paper, they move to the second stage of the algorithm as
        // soon as an `αₜ` is found which satisfies the conditions of
        // Theorem 3.1 (i.e. an `αₜ` for which `ψ(αₜ) <= 0` and `ψ'(αₜ) >= 0`).
        //
        // `ψ'(αₜ) = ф'(αₜ) - μ·ф'(0)`. So `ψ'(αₜ) >= 0` is equivalent to
        // `ф'(αₜ) >= μ·ф'(0)`. We replace this condition with a slightly
        // stronger one: `ф'(αₜ) >= min(μ, η)·ф'(0)`.
        auto const func_test =
            params.func_0 + state.t.alpha * params.f_tol * params.grad_0;
        if (first_stage && (state.t.func <= func_test)
            && (state.t.grad
                >= std::min(params.f_tol, params.g_tol) * params.grad_0)) {
            first_stage = false;
        }
        if (first_stage && (state.t.func <= state.x.func)
            && (state.t.func > func_test)) {
            LBFGS_TRACE("%s", "using modified updating scheme\n");
            with_modified_function_t context{state,
                                             params.f_tol * params.grad_0};
            update_trial_value_and_interval(state);
        }
        else {
            LBFGS_TRACE("%s", "using normal updating scheme\n");
            update_trial_value_and_interval(state);
        }
        ensure_shrinking();
    }
}
} // namespace detail

template <class Function>
auto line_search(
    Function&& value_and_gradient, ls_param_t const& params,
    double const alpha_0) noexcept(noexcept(std::
                                                declval<Function&>()(
                                                    std::declval<double>())))
    -> ls_result_t
{
    double cached = std::numeric_limits<double>::quiet_NaN();
    auto   r      = detail::line_search(
        [&value_and_gradient,
         &cached](auto const x) -> std::pair<double, double> {
            static_assert(std::is_same<decltype(x), double const>::value);
            cached                      = x;
            auto const [func, gradient] = value_and_gradient(x);
            return {func, gradient};
        },
        params, alpha_0);
    r.cached = cached == r.step;
    return r;
}

LBFGS_NAMESPACE_END

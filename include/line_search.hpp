
#pragma once

#include "config.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <tuple>
#include <utility>
#include <vector>

#include <unistd.h>
#include <cstdio>

LBFGS_NAMESPACE_BEGIN

enum class status_t {
    success = 0,
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

LBFGS_NAMESPACE_END

LBFGS_NAMESPACE_BEGIN

namespace detail {

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
    assert(f_a < f_b && "Precondition violated");
    assert((b - a) * df_a < 0.0 && "Precondition violated");
    auto const length = b - a;
    auto const scale  = (f_b - f_a) / length / df_a;
    auto const alpha  = a + 0.5 * length / (1.0 - scale);
    assert(std::min(a, b) < alpha && alpha < std::max(a, b)
           && "Postcondition violated");
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
    assert((df_b - df_a) * (b - a) > 0.0 && "Precondition violated");
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
#if 0
        std::cerr << "a = " << a << ", f(a) = " << f_a << ", f'(a) = " << df_a
                  << '\n'
                  << "b = " << b << ", f(b) = " << f_b << ", f'(b) = " << df_b
                  << '\n';
#endif
    assert(a != b && "Precondition violated");
    auto const length = b - a;
    auto const temp   = 3.0 * (f_a - f_b) / length + df_b;
    auto const theta  = temp + df_a;
    auto       gamma  = theta * theta - df_a * df_b;
    if (gamma >= 0.0) {
        gamma = std::sqrt(gamma);
        if (b < a) { gamma = -gamma; }
        auto const p = temp + gamma;
        auto const q = 2.0 * gamma + df_b - df_a;
#if 0
            std::cerr << "θ = " << theta
                      << /*", s = " << s <<*/ ", γ = " << gamma << ", p = " << p
                      << ", q = " << q << '\n';
#endif
        assert(q != 0.0);
        return a + p / q * length;
    }
    constexpr auto inf = std::numeric_limits<double>::infinity();
    return df_a > 0.0 ? -inf : inf;
}

struct interval_t {
    double min;
    double max;

    constexpr interval_t() noexcept : min{0.0}, max{0.0} {}

    constexpr interval_t(double const x, double const y, double const t,
                         bool const bracketed) noexcept
        : min{0.0}, max{0.0}
    {
        if (bracketed) { std::tie(min, max) = std::minmax(x, y); }
        else {
            min = x;
            max = t + 4.0 * (t - x);
            assert(min <= max);
        }
    }
};

struct endpoint_t {
    double alpha; ///< α
    double func;  ///< either ф(α) or ψ(α)
    double grad;  ///< either ф'(α) or ψ'(α)
};

struct state_t {
    endpoint_t x;         ///< Endpoint with the least function value
    endpoint_t y;         ///< The other endpoint
    endpoint_t t;         ///< Trial value
    interval_t interval;  ///< Interval for line search
    bool       bracketed; ///< Whether the trial value is bracketed
};

} // namespace detail

struct param_type {
    float    x_tol       = 1e-8f;
    float    f_tol       = 1e-4f;
    float    g_tol       = 1e-3f;
    float    step_min    = 1e-8f;
    float    step_max    = 1e8f;
    unsigned max_f_evals = 20;
};

namespace detail {
constexpr auto check_parameters(param_type const& p) noexcept -> status_t
{
    if (std::isnan(p.x_tol) || p.x_tol <= 0.0f)
        return status_t::invalid_interval_tolerance;
    if (std::isnan(p.f_tol) || p.f_tol <= 0.0f || p.f_tol >= 1.0f)
        return status_t::invalid_function_tolerance;
    if (std::isnan(p.g_tol) || p.g_tol <= 0.0f || p.g_tol >= 1.0f)
        return status_t::invalid_gradient_tolerance;
    if (std::isnan(p.step_min) || std::isnan(p.step_max) || p.step_min <= 0.0f
        || p.step_max <= p.step_min)
        return status_t::invalid_step_bounds;
    return status_t::success;
}

struct _internal_param_t {
    double   x_tol;
    double   f_tol;
    double   g_tol;
    double   step_min;
    double   step_max;
    unsigned max_f_evals;

    constexpr _internal_param_t(param_type const& p) noexcept
        : x_tol{static_cast<double>(p.x_tol)}
        , f_tol{static_cast<double>(p.f_tol)}
        , g_tol{static_cast<double>(p.g_tol)}
        , step_min{static_cast<double>(p.step_min)}
        , step_max{static_cast<double>(p.step_max)}
        , max_f_evals{p.max_f_evals}
    {
        assert(check_parameters(p) == status_t::success);
    }

    _internal_param_t()                         = delete;
    _internal_param_t(_internal_param_t const&) = delete;
    _internal_param_t(_internal_param_t&&)      = delete;
    _internal_param_t& operator=(_internal_param_t const&) = delete;
    _internal_param_t& operator=(_internal_param_t&&) = delete;
};

struct width_history_t {
    double previous;
    double current;

    constexpr auto push(double const new_width) noexcept -> void
    {
        assert(new_width >= 0 && "Width cannot be negative");
        assert(new_width <= current
               && "Width of the search interval should non-increasing");
        previous = current;
        current  = new_width;
    }
};

/// Case 1 on p. 299 of [1].
inline auto case_1(state_t const& state) noexcept
    -> std::tuple<double, bool, bool>
{
    auto const cubic = detail::minimise_cubic_interpolation(
        /*a=*/state.x.alpha, /*f_a=*/state.x.func, /*df_a=*/state.x.grad,
        /*b=*/state.t.alpha, /*f_b=*/state.t.func, /*df_b=*/state.t.grad);
    auto const quadratic = detail::minimise_quadratic_interpolation(
        /*a=*/state.x.alpha, /*f_a=*/state.x.func, /*df_a=*/state.x.grad,
        /*b=*/state.t.alpha, /*f_b=*/state.t.func);
    auto const alpha =
        std::abs(cubic - state.x.alpha) < std::abs(quadratic - state.x.alpha)
            ? cubic
            : cubic + 0.5 * (quadratic - cubic);
    LBFGS_TRACE("case_1: α_c=%f, α_q=%f -> α=%f\n", cubic, quadratic, alpha);
    return {alpha, /*bracketed=*/true, /*bound=*/true};
}

inline auto case_2(state_t const& state) noexcept
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
    LBFGS_TRACE("case_2: α_c=%f, α_s=%f -> α=%f\n", cubic, secant, alpha);
    return {alpha, /*bracketed=*/true, /*bound=*/false};
}

inline auto case_3(state_t const& state) noexcept
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
        && (cubic - state.t.alpha) * (state.t.alpha - state.x.alpha) >= 0.0) {
        static_assert(std::is_same_v<decltype(state.bracketed), bool>);
        auto const condition = state.bracketed
                               == (std::abs(cubic - state.t.alpha)
                                   < std::abs(secant - state.t.alpha));
        auto result = std::tuple{condition ? cubic : secant,
                                 /*bracketed=*/state.bracketed,
                                 /*bound=*/true};
        LBFGS_TRACE("case_3 (true): α_l=%f, α_t=%f, α_c=%f, α_s=%f -> α=%f\n",
                    state.x.alpha, state.t.alpha, cubic, secant,
                    std::get<0>(result));
        return result;
    }
    auto result =
        std::tuple{secant, /*bracketed=*/state.bracketed, /*bound=*/true};
    LBFGS_TRACE("case_3 (false): α_l=%f, α_t=%f, α_c=%f, α_s=%f -> α=%f\n",
                state.x.alpha, state.t.alpha, cubic, secant,
                std::get<0>(result));
    return result;
}

inline auto case_4(state_t const& state) noexcept
    -> std::tuple<double, bool, bool>
{
    auto const alpha =
        state.bracketed ? detail::minimise_cubic_interpolation(
            /*a=*/state.t.alpha, /*f_a=*/state.t.func,
            /*df_a=*/state.t.grad,
            /*b=*/state.y.alpha, /*f_b=*/state.y.func,
            /*df_b=*/state.y.grad)
                        : std::copysign(std::numeric_limits<float>::infinity(),
                                        state.t.alpha - state.x.alpha);
    LBFGS_TRACE("case_4: α=%f\n", alpha);
    return {alpha, /*bracketed=*/state.bracketed, /*bound=*/false};
}

inline auto handle_cases(state_t const& state) -> std::tuple<double, bool, bool>
{
    if (state.t.func > state.x.func) { return case_1(state); }
    if (state.x.grad * state.t.grad < 0.0) { return case_2(state); }
    // NOTE(twesterhout): The paper uses `<=` here!
    if (std::abs(state.t.grad) < std::abs(state.x.grad)) {
        return case_3(state);
    }
    return case_4(state);
}

inline auto cstep(state_t& state) -> void
{
    // Check the input parameters for errors.
    assert(!state.bracketed
           || (std::min(state.x.alpha, state.y.alpha) < state.t.alpha
               && state.t.alpha < std::max(state.x.alpha, state.y.alpha)));
    assert(state.x.grad * (state.t.alpha - state.x.alpha) < 0.0);
    assert(state.interval.min <= state.interval.max);

    bool   bound;
    double alpha;
    std::tie(alpha, state.bracketed, bound) = handle_cases(state);

    if (state.t.func > state.x.func) { state.y = state.t; }
    else {
        // if (state.t.grad != 0.0f) {
        if (state.x.grad * state.t.grad <= 0.0) { state.y = state.x; }
        state.x = state.t;
        // }
    }
    LBFGS_TRACE("cstep: new α_l=%f, α_u=%f\n", state.x.alpha, state.y.alpha);
    // state.interval = interval_t{state.x.alpha, state.y.alpha, state.t.alpha,
    //                             state.bracketed};
    alpha = std::clamp(alpha, state.interval.min, state.interval.max);
    if (state.bracketed && bound) {
        auto const middle =
            state.x.alpha + 0.66 * (state.y.alpha - state.x.alpha);
        LBFGS_TRACE("cstep: bracketed && bound: α=%f, middle=%f\n", alpha,
                    middle);
        alpha = (state.x.alpha < state.y.alpha) ? std::min(middle, alpha)
                                                : std::max(middle, alpha);
    }
    state.t.alpha = alpha;
    state.t.func  = std::numeric_limits<double>::quiet_NaN();
    state.t.grad  = std::numeric_limits<double>::quiet_NaN();
}
} // namespace detail

struct result_t {
    status_t status;
    float    step;
    float    func;
    float    grad;
    unsigned num_f_evals;
};

namespace detail {
struct _internal_result_t {
    status_t status;
    double   step;
    double   func;
    double   grad;
    unsigned num_f_evals;

    operator result_t() const noexcept
    {
        return {status, static_cast<float>(step), static_cast<float>(func),
                static_cast<float>(grad), num_f_evals};
    }
};

#define TCM_DEBUG

template <class Function>
auto line_search(Function value_and_gradient, _internal_param_t const& params,
                 double const func_0, double const grad_0, double const alpha_0)
    -> _internal_result_t
{
    constexpr auto NaN = std::numeric_limits<double>::quiet_NaN();
    assert(grad_0 < 0);
    assert(alpha_0 > 0);

    // TODO(twesterhout): Explain
    state_t state = state_t{
        /*x=*/endpoint_t{/*alpha=*/0, /*func=*/func_0, /*grad=*/grad_0},
        /*y=*/endpoint_t{/*alpha=*/0, /*func=*/func_0, /*grad=*/grad_0},
        /*t=*/endpoint_t{/*alpha=*/alpha_0, /*func=*/NaN, /*grad=*/NaN},
        /*interval=*/interval_t{0, 0, alpha_0, false},
        /*bracketed=*/false,
    };

    auto const grad_test   = params.f_tol * grad_0;
    auto       num_f_evals = 0u;   // Current number of function evaluations
    auto       first_stage = true; // TODO(twesterhout): Explain stages

    width_history_t width_history;
    width_history.current  = params.step_max - params.step_min;
    width_history.previous = 2 * width_history.current;

    for (;;) {
        state.interval = interval_t{state.x.alpha, state.y.alpha, state.t.alpha,
                                    state.bracketed};
        // Force the step to be within the bounds stpmax and stpmin.
        state.t.alpha =
            std::clamp(state.t.alpha, static_cast<double>(params.step_min),
                       static_cast<double>(params.step_max));

        LBFGS_TRACE("proposed αₜ=%f ∈ [%f, %f]\n", state.t.alpha,
                    state.interval.min, state.interval.max);

        // Search interval has shrunk below the threshold. We reset current
        // position to the best one obtained so far and stop.
        if (state.bracketed
            && (state.interval.max - state.interval.min
                <= params.x_tol * state.interval.max)) {
            LBFGS_TRACE("interval too small: %f <= %f",
                        state.interval.max - state.interval.min,
                        params.x_tol * state.interval.max);
            // NOTE: We return `state.x.alpha` rather than `state.t.alpha`!
            return {status_t::interval_too_small, state.x.alpha, state.x.func,
                    state.x.grad, num_f_evals};
        }

        // We reached the maximum number of evaluations of `f`. Even if we
        // come up with a guess for a better step size, we can't check it.
        // So we use the best position step size obtained so far.
        assert(num_f_evals <= params.max_f_evals);
        if (num_f_evals == params.max_f_evals) {
            LBFGS_TRACE("too many function evaluations: %u == %u\n",
                        num_f_evals, params.max_f_evals);
            // NOTE: We return `state.x.alpha` rather than `state.t.alpha`!
            return {status_t::too_many_function_evaluations, state.x.alpha,
                    state.x.func, state.x.grad, num_f_evals};
        }

        // TODO(twesterhout): Handle this properly!
        if (state.bracketed && state.t.alpha == state.interval.min) {
            return {status_t::rounding_errors_prevent_progress, state.x.alpha,
                    state.x.func, state.x.grad, num_f_evals};
        }
        assert(!state.bracketed
               || (state.interval.min < state.t.alpha
                   && state.t.alpha < state.interval.max));

        std::tie(state.t.func, state.t.grad) =
            value_and_gradient(state.t.alpha);
        ++num_f_evals;
        auto const func_test = func_0 + state.t.alpha * grad_test;
        LBFGS_TRACE("another function evaluation: αₜ=%f, ф(αₜ)=%f, ф'(αₜ)=%f\n",
                    state.t.alpha, state.t.func, state.t.grad);
        // (p.292 of J.J. Moré & D.C. Thuente 1994):

        // First case in Theorem 2.2 (p. 292): we reached `αₘₐₓ` and
        // condition (2.3) holds (i.e. `ψ(αₜ) <= 0` and `ψ'(αₜ) < 0`).
        if (state.t.alpha == params.step_max && state.t.func <= func_test
            && state.t.grad <= grad_test) {
            LBFGS_TRACE("reached αₘₐₓ: ф'(αₜ)=%f <= %f\n", state.t.grad,
                        grad_test);
            return {status_t::maximum_step_reached, state.t.alpha, state.t.func,
                    state.t.grad, num_f_evals};
        }

        // Second case in Theorem 2.2 (p. 292): we reached `αₘᵢₙ` and
        // condition (2.4) holds (i.e. `ψ(αₜ) > 0` and `ψ'(αₜ) >= 0`).
        if (state.t.alpha == params.step_min
            && (state.t.func > func_test || state.t.grad >= grad_test)) {
            LBFGS_TRACE("reached αₘᵢₙ: ф'(αₜ)=%f >= %f\n", state.t.grad,
                        grad_test);
            return {status_t::minimum_step_reached, state.t.alpha, state.t.func,
                    state.t.grad, num_f_evals};
        }

        // Both sufficient decrease (1.1) and curvatuve (1.2) conditions
        // hold (i.e. `ψ(αₜ) <= 0` and `|ф'(αₜ)| <= η·|ф'(0)|`). We use the
        // fact that `|ф'(0)| = -ф'(0)`, because `ф'(0) < 0` is a
        // precondition for the whole algorithm.
        if ((state.t.func <= func_test)
            && (std::abs(state.t.grad) <= params.g_tol * (-grad_0))) {
            LBFGS_TRACE("Strong Wolfe conditions satisfied:\n"
                        "    sufficient decrease: %f <= %f\n"
                        "    curvature condition: %f <= %f\n"
                        "    %u function evaluations\n",
                        state.t.func, func_test, std::abs(state.t.grad),
                        params.g_tol * (-grad_0), num_f_evals);
            return {status_t::success, state.t.alpha, state.t.func,
                    state.t.grad, num_f_evals};
        }

        // In the paper, they move to the second stage of the algorithm as
        // soon as an `αₜ` is found which satisfies the conditions of
        // Theorem 3.1 (i.e. an `αₜ` for which `ψ(αₜ) <= 0` and `ψ'(αₜ) >= 0`).
        //
        // `ψ'(αₜ) = ф'(αₜ) - μ·ф'(0)`. So `ψ'(αₜ) >= 0` is equivalent to
        // `ф'(αₜ) >= μ·ф'(0)`. We replace this condition with a slightly
        // stronger one: `ф'(αₜ) >= min(μ, η)·ф'(0)`.
        if (first_stage && (state.t.func <= func_test)
            && (state.t.grad
                >= std::min(params.f_tol, params.g_tol) * grad_0)) {
            first_stage = false;
        }

        if (first_stage && (state.t.func <= state.x.func)
            && (state.t.func > func_test)) {

            LBFGS_TRACE("%s", "using modified updating scheme\n");
            struct use_modified_function {
                state_t&     _s;
                double const _grad_test;

                constexpr use_modified_function(state_t&     s,
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

                ~use_modified_function()
                {
                    _s.x.func += _s.x.alpha * _grad_test;
                    _s.x.grad += _grad_test;
                    _s.y.func += _s.y.alpha * _grad_test;
                    _s.y.grad += _grad_test;
                    assert(std::isnan(_s.t.func));
                    assert(std::isnan(_s.t.grad));
                }
            } dummy{state, grad_test};
            cstep(state);
        }
        else {
            LBFGS_TRACE("%s", "using normal updating scheme\n");
            cstep(state);
        }

        if (state.bracketed) {
            // Last paragraph of p.292:
            // <quote>If the algorithm generates an interval I in [αₘᵢₙ,
            // αₘₐₓ], then we need a third rule to guarantee that the choice
            // of αₜ forces the length of I to zero. In our implementation
            // this is done by monitoring the length of I; if the length of
            // I does not decrease by a factor δ < 1 after two trials, then
            // a bisection step is used for the next trial αₜ.</quote>
            constexpr auto delta     = 0.66;
            auto const     new_width = std::abs(state.y.alpha - state.x.alpha);
            if (new_width >= delta * width_history.previous) {
                state.t.alpha =
                    state.x.alpha + 0.5 * (state.y.alpha - state.x.alpha);
            }
            width_history.push(new_width);
        }
    }
}
} // namespace detail

template <class Function>
auto line_search(Function&& value_and_gradient, param_type const& params,
                 float const func_0, float const grad_0,
                 float const alpha_0 = 1.0f) -> result_t
{
    auto wrapper = [&value_and_gradient](double const x) {
        float func, gradient;
        std::tie(func, gradient) = value_and_gradient(static_cast<float>(x));
        return std::make_pair(static_cast<double>(func),
                              static_cast<double>(gradient));
    };
    auto result = detail::line_search(
        std::move(wrapper), detail::_internal_param_t{params},
        static_cast<double>(func_0), static_cast<double>(grad_0),
        static_cast<double>(alpha_0));
    return result;
}

LBFGS_NAMESPACE_END

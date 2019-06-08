
#include "line_search.hpp"
#include <system_error>

LBFGS_NAMESPACE_BEGIN

namespace { // anonymous namespace
struct lbfgs_error_category : public std::error_category {
    auto name() const noexcept -> char const* override;
    auto message(int const value) const -> std::string override;
    ~lbfgs_error_category() override = default;
    static auto instance() noexcept -> std::error_category const&;
};

auto lbfgs_error_category::name() const noexcept -> char const*
{
    return "lbfgs category";
}

auto lbfgs_error_category::message(int const value) const -> std::string
{
    switch (static_cast<status_t>(value)) {
    case status_t::success: return "no error";
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
    default: return "(unrecognised error)";
    } // end switch
}

auto lbfgs_error_category::instance() noexcept -> std::error_category const&
{
    static lbfgs_error_category c;
    return c;
}
} // namespace

LBFGS_EXPORT auto make_error_code(status_t const e) noexcept -> std::error_code
{
    return {static_cast<int>(e), lbfgs_error_category::instance()};
}

namespace detail {

[[noreturn]] LBFGS_EXPORT auto assert_fail(char const* expr, char const* file,
                                           unsigned line, char const* function,
                                           char const* msg) noexcept -> void
{
    std::fprintf(stderr,
                 LBFGS_BUG_MESSAGE
                 "\n\x1b[1m\x1b[91mAssertion failed\x1b[0m at %s:%u: %s: "
                 "\"\x1b[1m\x1b[97m%s\x1b[0m\" evaluated to false: "
                 "\x1b[1m\x1b[97m%s\x1b[0m\n",
                 file, line, function, expr, msg);
    std::terminate();
}

namespace {
    /// \brief Case 1 on p. 299 of [1].
    ///
    /// \return `(αₜ⁺, bracketed, bound)` where `αₜ⁺` is the trial value in the new
    /// search interval `I⁺`.
    inline auto case_1(state_t const& state) noexcept
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
        LBFGS_TRACE("case_1: α_c=%f, α_q=%f -> α=%f\n", cubic, quadratic,
                    alpha);
        return {alpha, /*bracketed=*/true, /*bound=*/true};
    }

    /// \brief Case 2 on p. 299 of [1].
    ///
    /// \return `(αₜ⁺, bracketed, bound)` where `αₜ⁺` is the trial value in the new
    /// search interval `I⁺`.
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
            && (cubic - state.t.alpha) * (state.t.alpha - state.x.alpha)
                   >= 0.0) {
            static_assert(std::is_same_v<decltype(state.bracketed), bool>);
            auto const condition = state.bracketed
                                   == (std::abs(cubic - state.t.alpha)
                                       < std::abs(secant - state.t.alpha));
            auto result = std::tuple{condition ? cubic : secant,
                                     /*bracketed=*/state.bracketed,
                                     /*bound=*/true};
            LBFGS_TRACE(
                "case_3 (true): α_l=%f, α_t=%f, α_c=%f, α_s=%f -> α=%f\n",
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
            state.bracketed
                ? detail::minimise_cubic_interpolation(
                    /*a=*/state.t.alpha, /*f_a=*/state.t.func,
                    /*df_a=*/state.t.grad,
                    /*b=*/state.y.alpha, /*f_b=*/state.y.func,
                    /*df_b=*/state.y.grad)
                : std::copysign(std::numeric_limits<float>::infinity(),
                                state.t.alpha - state.x.alpha);
        LBFGS_TRACE("case_4: α=%f\n", alpha);
        return {alpha, /*bracketed=*/state.bracketed, /*bound=*/false};
    }

    inline auto handle_cases(state_t const& state) noexcept
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

LBFGS_EXPORT auto update_trial_value_and_interval(state_t& state) noexcept
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
    LBFGS_TRACE("cstep: new α_l=%f, α_u=%f\n", state.x.alpha, state.y.alpha);
    alpha = std::clamp(alpha, state.interval.min(), state.interval.max());
    if (state.bracketed && bound) {
        auto const middle =
            state.x.alpha + 0.66 * (state.y.alpha - state.x.alpha);
        LBFGS_TRACE("cstep: bracketed && bound: α=%f, middle=%f\n", alpha,
                    middle);
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

LBFGS_NAMESPACE_END

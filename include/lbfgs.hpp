
#pragma once

#include "config.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <numeric>
#include <optional>
#include <system_error>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <gsl/gsl-lite.hpp>
#include <cblas.h>

#include <unistd.h>
#include <cstdio>

#include "line_search.hpp"

LBFGS_NAMESPACE_BEGIN

struct lbfgs_param_t {
    /// Number of vectors used for representing the inverse Hessian matrix.
    unsigned m = 6;
    /// Distance for delta-based convergence tests.
    unsigned past = 0;
    /// Maximum number of BFGS iterations to perform.
    unsigned max_iter = 0;
    /// Parameter ε for convergence test: `‖∇f(x)‖₂ < ε·max(1, ‖x‖₂)`.
    double epsilon = 1e-5;
    /// Parameter δ for convergence test: |f(x - past) - f| < δ·|f|
    double delta = 1e-5;
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

namespace detail {

// A hacky way of determining the integral type BLAS uses for sizes and
// increments: we pattern match on the signature of `cblas_sdot`.
template <class T> struct get_blas_int_type;

template <class T>
struct get_blas_int_type<float (*)(T, float const*, T, float const*, T)> {
    using type = T;
};

using blas_int = typename get_blas_int_type<decltype(&cblas_sdot)>::type;

inline auto dot(gsl::span<float const> a, gsl::span<float const> b) noexcept
    -> double
{
    LBFGS_ASSERT(a.size() == b.size(), "incompatible dimensions");
    LBFGS_ASSERT(
        a.size() <= static_cast<size_t>(std::numeric_limits<blas_int>::max()),
        "integer overflow");
    return cblas_dsdot(static_cast<blas_int>(a.size()), a.data(), 1, b.data(),
                       1);
}

inline auto nrm2(gsl::span<float const> x) noexcept -> double
{
    LBFGS_ASSERT(
        x.size() <= static_cast<size_t>(std::numeric_limits<blas_int>::max()),
        "integer overflow");
#if defined(LBFGS_CLANG)
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wdouble-promotion"
#endif
    return cblas_snrm2(static_cast<blas_int>(x.size()), x.data(), 1);
#if defined(LBFGS_CLANG)
#    pragma clang diagnostic pop
#endif
}

inline auto axpy(float const a, gsl::span<float const> x,
                 gsl::span<float> y) noexcept -> void
{
    LBFGS_ASSERT(x.size() == y.size(), "incompatible dimensions");
    LBFGS_ASSERT(
        x.size() <= static_cast<size_t>(std::numeric_limits<blas_int>::max()),
        "integer overflow");
    cblas_saxpy(static_cast<blas_int>(x.size()), a, x.data(), 1, y.data(), 1);
}

inline auto scal(float const a, gsl::span<float> x) noexcept -> void
{
    LBFGS_ASSERT(
        x.size() <= static_cast<size_t>(std::numeric_limits<blas_int>::max()),
        "integer overflow");
    cblas_sscal(static_cast<blas_int>(x.size()), a, x.data(), 1);
}

inline auto negative_copy(gsl::span<float const> const src,
                          gsl::span<float> const       dst) noexcept -> void
{
    LBFGS_ASSERT(src.size() == dst.size(), "incompatible dimensions");
    for (auto i = size_t{0}; i < src.size(); ++i) {
        dst[i] = -src[i];
    }
}
} // namespace detail

struct iteration_data_t {
    float            s_dot_y;
    float            alpha;
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

    iteration_history_t(iteration_history_t const&) = delete;
    iteration_history_t(iteration_history_t&&)      = delete;
    iteration_history_t& operator=(iteration_history_t const&) = delete;
    iteration_history_t& operator=(iteration_history_t&&) = delete;

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

template <size_t Alignment>
constexpr auto align_up(size_t const value) noexcept -> size_t
{
    static_assert(Alignment != 0 && (Alignment & (Alignment - 1)) == 0,
                  "Invalid alignment");
    return (value + (Alignment - 1)) & ~(Alignment - 1);
}

struct lbfgs_point_t {
    double           value;
    gsl::span<float> x;
    gsl::span<float> grad;

    constexpr lbfgs_point_t(double const _value, gsl::span<float> _x,
                            gsl::span<float> _grad) noexcept
        : value{_value}, x{_x}, grad{_grad}
    {}

    lbfgs_point_t(lbfgs_point_t const& other) = delete;
    lbfgs_point_t(lbfgs_point_t&&)            = delete;

    lbfgs_point_t& operator=(lbfgs_point_t const& other) noexcept
    {
        value = other.value;
        // TODO: Check that ranges don't overlap.
        std::copy(std::begin(other.x), std::end(other.x), std::begin(x));
        std::copy(std::begin(other.grad), std::end(other.grad),
                  std::begin(grad));
        return *this;
    }
};

struct lbfgs_state_t {
    iteration_history_t history;
    lbfgs_point_t       current;
    lbfgs_point_t       previous;
    gsl::span<float>    direction;
};

struct lbfgs_buffers_t {
  private:
    std::vector<float>            _workspace;
    std::vector<iteration_data_t> _history;
    size_t                        _n;

    static constexpr auto number_vectors(size_t const m) noexcept -> size_t
    {
        return 2 * m /* s and y vectors of the last m iterations */
               + 1   /* x */
               + 1   /* x_prev */
               + 1   /* g */
               + 1   /* g_prev */
               + 1;  /* d */
    }

    static constexpr auto vector_size(size_t const n) noexcept -> size_t
    {
        return align_up<64UL / sizeof(float)>(n);
    }

    auto get(size_t const i) noexcept -> gsl::span<float>
    {
        LBFGS_ASSERT((i + 1) * _n <= _workspace.size(), "index out of bounds");
        auto const size = vector_size(_n);
        return {_workspace.data() + i * size, _n};
    }

  public:
    lbfgs_buffers_t(size_t const n, size_t const m);
    auto resize(size_t n, size_t const m) -> void;
    auto make_state() noexcept -> lbfgs_state_t;
};

inline auto print_span(char const* prefix, gsl::span<float const> xs) -> void
{
    std::printf("%s [", prefix);
    if (!xs.empty()) {
        std::printf("%f", xs[0]);
        for (auto i = size_t{1}; i < xs.size(); ++i) {
            std::printf(", %f", xs[i]);
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

        auto operator()(double const alpha) const noexcept(is_noexcept())
        {
            for (auto i = size_t{0}; i < x.size(); ++i) {
                x[i] = x_0[i] + static_cast<float>(alpha) * direction[i];
            }
            // Yes, we want implicit conversion to double here!
            double const f_x = value_and_gradient(
                static_cast<gsl::span<float const>>(x), grad);
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
        if (result.status != status_t::success) { return result.status; }
        // TODO(twesterhout): This runs an extra dot operation
        if (!result.cached) { wrapper(result.step); }
        _state.current.value = result.func;
        LBFGS_TRACE("f(x)=%f, f'(x)=%f\n", _state.current.value,
                    static_cast<double>(result.grad));
        print_span("x = ", _state.current.x);
        return status_t::success;
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

template <class Function>
auto minimize(Function value_and_gradient, lbfgs_param_t const& params,
              lbfgs_state_t& state) -> status_t
{
    state.current.value = value_and_gradient(
        gsl::span<float const>{state.current.x}, state.current.grad);

    gradient_small_enough_fn gradient_is_small{state, params};

    auto const grad_0_norm = detail::nrm2(state.current.grad);
    if (gradient_is_small(grad_0_norm)) { return status_t::success; }
    auto step = 1.0 / grad_0_norm;

    detail::negative_copy(state.current.grad, state.direction);

    line_search_runner_fn do_line_search{state, params};

    for (auto iteration = 1u;; ++iteration) {
        print_span("grad = ", state.current.grad);
        state.previous = state.current;
        print_span("d = ", state.direction);

        if (auto const status = do_line_search(value_and_gradient, step);
            status != status_t::success) {
            return status;
        }

        if (gradient_is_small()) { return status_t::success; }
        if (iteration == params.max_iter) {
            return status_t::too_many_iterations;
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

LBFGS_NAMESPACE_END

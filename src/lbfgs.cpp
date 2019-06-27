
#include "lbfgs.hpp"
#include <system_error>

LBFGS_NAMESPACE_BEGIN

namespace { // anonymous namespace
struct lbfgs_error_category : public std::error_category {
    constexpr lbfgs_error_category() noexcept = default;

    lbfgs_error_category(lbfgs_error_category const&) = delete;
    lbfgs_error_category(lbfgs_error_category&&)      = delete;
    lbfgs_error_category& operator=(lbfgs_error_category const&) = delete;
    lbfgs_error_category& operator=(lbfgs_error_category&&) = delete;

    ~lbfgs_error_category() override = default;

    [[nodiscard]] auto name() const noexcept -> char const* override;
    [[nodiscard]] auto message(int value) const -> std::string override;
    static auto        instance() noexcept -> std::error_category const&;
};

auto lbfgs_error_category::name() const noexcept -> char const*
{
    return "lbfgs category";
}

auto lbfgs_error_category::message(int const value) const -> std::string
{
    switch (static_cast<status_t>(value)) {
    case status_t::success: return "no error";
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

namespace {
    /// \brief Case 1 on p. 299 of [1].
    ///
    /// \return `(αₜ⁺, bracketed, bound)` where `αₜ⁺` is the trial value in the new
    /// search interval `I⁺`.
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
        LBFGS_TRACE("case_1: α_c=%f, α_q=%f -> α=%f\n", cubic, // NOLINT
                    quadratic,                                 // NOLINT
                    alpha);                                    // NOLINT
        return {alpha, /*bracketed=*/true, /*bound=*/true};
    }

    /// \brief Case 2 on p. 299 of [1].
    ///
    /// \return `(αₜ⁺, bracketed, bound)` where `αₜ⁺` is the trial value in the new
    /// search interval `I⁺`.
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
        LBFGS_TRACE("case_2: α_c=%f, α_s=%f -> α=%f\n", cubic, secant, // NOLINT
                    alpha);                                            // NOLINT
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
            LBFGS_TRACE( // NOLINT
                "case_3 (true): α_l=%f, α_t=%f, α_c=%f, α_s=%f -> α=%f\n", // NOLINT
                state.x.alpha, state.t.alpha, cubic, secant, // NOLINT
                std::get<0>(result));                        // NOLINT
            return result;
        }
        auto result =
            std::tuple{secant, /*bracketed=*/state.bracketed, /*bound=*/true};
        LBFGS_TRACE( // NOLINT
            "case_3 (false): α_l=%f, α_t=%f, α_c=%f, α_s=%f -> α=%f\n", // NOLINT
            state.x.alpha, state.t.alpha, cubic, secant, // NOLINT
            std::get<0>(result));                        // NOLINT
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

namespace detail {
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
#if defined(LBFGS_CLANG)
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wdouble-promotion"
#endif
    return cblas_snrm2(static_cast<blas_int>(x.size()), x.data(), 1);
#if defined(LBFGS_CLANG)
#    pragma clang diagnostic pop
#endif
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

} // namespace detail

LBFGS_EXPORT lbfgs_buffers_t::lbfgs_buffers_t() noexcept
    : _workspace{}, _history{}, _func_history{}, _n{}
{}

LBFGS_EXPORT lbfgs_buffers_t::lbfgs_buffers_t(size_t const n, size_t const m,
                                              size_t const past)
    : _workspace{}, _history{}, _func_history{}, _n{}
{
    resize(n, m, past);
}

constexpr auto lbfgs_buffers_t::number_vectors(size_t const m) noexcept
    -> size_t
{
    return 2 * m /* s and y vectors of the last m iterations */
           + 1   /* x */
           + 1   /* x_prev */
           + 1   /* g */
           + 1   /* g_prev */
           + 1;  /* d */
}

constexpr auto lbfgs_buffers_t::vector_size(size_t const n) noexcept -> size_t
{
    return align_up<64UL / sizeof(float)>(n);
}

auto lbfgs_buffers_t::get(size_t const i) noexcept -> gsl::span<float>
{
    auto const size = vector_size(_n);
    LBFGS_TRACE("%zu * %zu + %zu <= %zu\n", i, size, _n, _workspace.size());
    LBFGS_ASSERT(i * size + _n <= _workspace.size(), "index out of bounds");
    return {_workspace.data() + i * size, _n};
}

LBFGS_EXPORT auto lbfgs_buffers_t::resize(size_t const n, size_t const m,
                                          size_t const past) -> void
{
    if (n != _n || m != _history.size()) {
        // Since _workspace may need to be re-allocated, we don't want to
        // keep dangling pointers
        _history.clear();
        _history.resize(
            m, {0.0, std::numeric_limits<double>::quiet_NaN(), {}, {}});
        _n = n;
        _workspace.resize(vector_size(n) * number_vectors(m));
        for (auto i = size_t{0}; i < _history.size(); ++i) {
            _history[i].s = get(2 * i);
            _history[i].y = get(2 * i + 1);
        }
    }
    if (past != _func_history.size()) { _func_history.resize(past); }
}

LBFGS_EXPORT auto lbfgs_buffers_t::make_state() noexcept -> lbfgs_state_t
{
    constexpr auto NaN = std::numeric_limits<double>::quiet_NaN();
    auto const     m   = _history.size();
    return lbfgs_state_t{{gsl::span<iteration_data_t>{_history}},
                         {NaN, get(2 * m + 0), get(2 * m + 1)},
                         {NaN, get(2 * m + 2), get(2 * m + 3)},
                         get(2 * m + 4),
                         {gsl::span<double>{_func_history}}};
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

constexpr auto iteration_history_t::emplace_back_impl(
    gsl::span<float const> x, gsl::span<float const> x_prev,
    gsl::span<float const> g, gsl::span<float const> g_prev) noexcept -> double
{
    auto idx = back_index();
    if (_size == capacity()) { _first = sum(_first, 1); }
    else {
        ++_size;
    }

    // TODO: Optimise this loop
    auto&      s       = _data[idx].s;
    auto&      y       = _data[idx].y;
    auto const n       = s.size();
    auto       s_dot_y = 0.0;
    auto       y_dot_y = 0.0;
    for (auto i = size_t{0}; i < n; ++i) {
        auto const s_i =
            static_cast<double>(x[i]) - static_cast<double>(x_prev[i]);
        auto const y_i =
            static_cast<double>(g[i]) - static_cast<double>(g_prev[i]);
        s[i] = static_cast<float>(s_i);
        y[i] = static_cast<float>(y_i);
        s_dot_y += s_i * y_i;
        y_dot_y += y_i * y_i;
    }
    _data[idx].s_dot_y = s_dot_y;
    _data[idx].alpha   = std::numeric_limits<double>::quiet_NaN();
    LBFGS_ASSERT(s_dot_y > 0, "something went wrong during line search");
    return s_dot_y / y_dot_y;
}

LBFGS_EXPORT auto iteration_history_t::emplace_back(
    gsl::span<float const> x, gsl::span<float const> x_prev,
    gsl::span<float const> g, gsl::span<float const> g_prev) noexcept -> double
{
    return emplace_back_impl(x, x_prev, g, g_prev);
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
    r -= (r >= capacity()) * capacity();
    return r;
}

constexpr auto iteration_history_t::back_index() const noexcept -> size_type
{
    return sum(_first, _size);
}

template <bool IsConst> class iteration_history_t::history_iterator {
  public:
    using type            = history_iterator<IsConst>;
    using value_type      = iteration_data_t;
    using difference_type = std::ptrdiff_t;
    using pointer = std::conditional_t<IsConst, value_type const, value_type>*;
    using reference =
        std::conditional_t<IsConst, value_type const, value_type>&;
    using iterator_category = std::bidirectional_iterator_tag;

    constexpr history_iterator() noexcept                        = default;
    constexpr history_iterator(history_iterator const&) noexcept = default;
    constexpr history_iterator(history_iterator&&) noexcept      = default;
    constexpr history_iterator&
    operator=(history_iterator const&) noexcept = default;
    constexpr history_iterator&
    operator=(history_iterator&&) noexcept = default;

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
                      print_span("q=", q);
                  });
    // r <- H_k^0*q
    detail::scal(static_cast<float>(gamma), q);
    printf("γ=%f\n", gamma);
    print_span("q=", q);
    //for i = k − m, k − m + 1, . . . , k − 1
    std::for_each(begin, end, [q](auto& x) {
        // beta <- rho_i * y_i^T * r
        auto const beta = detail::dot(x.y, q) / x.s_dot_y;
        // r <- r + s_i * ( alpha_i - beta)
        detail::axpy(static_cast<float>(x.alpha - beta), x.s, q);
        printf("β=%f\n", beta);
        print_span("q=", q);
    });
    // stop with result "H_k*f_f'=q"
}

LBFGS_EXPORT auto apply_inverse_hessian(iteration_history_t&   history,
                                        double const           gamma,
                                        gsl::span<float> const q) -> void
{
    apply_inverse_hessian(history.begin(), history.end(), gamma, q);
}

LBFGS_NAMESPACE_END


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
#include <SG14/ring.h>
#include <cblas.h>

#include <unistd.h>
#include <cstdio>

#include "line_search.hpp"

extern "C" {
double cblas_dsdot(int32_t n, const float* sx, int32_t incx, const float* sy,
                   int32_t incy);
void   cblas_saxpy(int32_t n, float a, const float* x, const int32_t incx,
                   float* y, int32_t incy);
void   cblas_sscal(int32_t n, float a, float* x, int32_t incx);
} // extern "C"

LBFGS_NAMESPACE_BEGIN

namespace detail {
inline auto dot(gsl::span<float const> a, gsl::span<float const> b) noexcept
    -> double
{
    LBFGS_ASSERT(a.size() == b.size(), "incompatible dimensions");
    LBFGS_ASSERT(
        a.size() <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
        "integer overflow");
    return cblas_dsdot(static_cast<size_t>(a.size()), a.data(), 1, b.data(), 1);
}

inline auto nrm2(gsl::span<float const> x) noexcept -> double
{
    LBFGS_ASSERT(
        x.size() <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
        "integer overflow");
    return cblas_snrm2(static_cast<size_t>(x.size()), x.data(), 1);
}

inline auto axpy(float const a, gsl::span<float const> x,
                 gsl::span<float> y) noexcept -> void
{
    LBFGS_ASSERT(x.size() == y.size(), "incompatible dimensions");
    LBFGS_ASSERT(
        x.size() <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
        "integer overflow");
    cblas_saxpy(static_cast<int32_t>(x.size()), a, x.data(), 1, y.data(), 1);
}

inline auto scal(float const a, gsl::span<float> x) noexcept -> void
{
    LBFGS_ASSERT(
        x.size() <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
        "integer overflow");
    cblas_sscal(static_cast<int32_t>(x.size()), a, x.data(), 1);
}

inline auto axpby(float const a, gsl::span<float const> x, float const b,
                  gsl::span<float const> y, gsl::span<float> out) noexcept
    -> void
{
    LBFGS_ASSERT(x.size() == y.size() && y.size() == out.size(),
                 "incompatible dimensions");
    LBFGS_ASSERT(
        x.size() <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
        "integer overflow");
    for (auto i = size_t{0}; i < x.size(); ++i) {
        out[i] = a * x[i] + b * y[i];
    }
}
} // namespace detail

struct lbfgs_param_t {
    unsigned   m        = 5;
    float      epsilon  = 1e-5;
    unsigned   past     = 0;
    float      delta    = 1e-5;
    unsigned   max_iter = 0;
    param_type line_search;

    constexpr lbfgs_param_t() noexcept
        : m{5}, epsilon{1e-5}, past{0}, delta{1e-5}, max_iter{0}, line_search{}
    {}
};

struct iteration_data_t {
    float            s_dot_y;
    float            alpha;
    gsl::span<float> s;
    gsl::span<float> y;
};

class iteration_history_t {

    static_assert(std::is_nothrow_copy_assignable_v<iteration_data_t>);
    static_assert(std::is_nothrow_move_assignable_v<iteration_data_t>);
    template <bool> class history_iterator;

  public:
    using size_type      = size_t;
    using iterator       = history_iterator<false>;
    using const_iterator = history_iterator<true>;

    /// Constructs an empty history object.
    constexpr iteration_history_t(gsl::span<iteration_data_t> data) noexcept
        : _first{0}, _size{0}, _data{data}
    {}

    iteration_history_t(iteration_history_t const&) = delete;
    iteration_history_t(iteration_history_t&&)      = delete;
    iteration_history_t& operator=(iteration_history_t const&) = delete;
    iteration_history_t& operator=(iteration_history_t&&) = delete;

    constexpr auto capacity() const noexcept { return _data.size(); }
    constexpr auto size() const noexcept { return _size; }
    constexpr auto empty() const noexcept { return _size == 0; }
    constexpr auto full() const noexcept { return size() == capacity(); }

    constexpr auto push_back(iteration_data_t const& x) noexcept -> void
    {
        _data[back_index()] = x;
        if (++_size > capacity()) { _first = sum(_first, 1); }
    }

    constexpr auto emplace_back(gsl::span<float const> x,
                                gsl::span<float const> x_prev,
                                gsl::span<float const> g,
                                gsl::span<float const> g_prev) noexcept -> float
    {
        auto idx = back_index();
        if (++_size > capacity()) { _first = sum(_first, 1); }
        auto&      s       = _data[idx].s;
        auto&      y       = _data[idx].y;
        auto const n       = s.size();
        auto       s_dot_y = 0.0;
        auto       y_dot_y = 0.0;
        for (auto i = size_t{0}; i < n; ++i) {
            s[i] = x[i] - x_prev[i];
            y[i] = g[i] - g_prev[i];
            s_dot_y += s[i] * y[i];
            y_dot_y += s[i] * y[i];
        }
        _data[idx].s_dot_y = s_dot_y;
        _data[idx].alpha   = std::numeric_limits<float>::quiet_NaN();
        return s_dot_y / y_dot_y;
    }

  private:
    template <bool IsConst> class history_iterator {
      public:
        using type            = history_iterator<IsConst>;
        using value_type      = iteration_data_t;
        using difference_type = std::ptrdiff_t;
        using pointer =
            std::conditional_t<IsConst, value_type const, value_type>*;
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
            LBFGS_ASSERT(_obj != nullptr && _obj->is_valid_index(_i),
                         "iterator not dereferenceable");
            return _obj->_data[_i];
        }

        constexpr auto operator-> () const noexcept -> pointer
        {
            return std::addressof(*(*this));
        }

        constexpr auto operator++() noexcept -> type&
        {
            LBFGS_ASSERT(_obj != nullptr && _obj->is_valid_index(_i),
                         "iterator not incrementable");
            ++_i;
            if (_i == _obj->capacity()) { _i = 0; }
            return *this;
        }

        constexpr auto operator++(int) noexcept -> type&
        {
            auto temp{*this};
            ++(*this);
            return temp;
        }

        constexpr auto operator--() noexcept -> type&
        {
            if (_i == 0) { _i = _obj->capacity() - 1; }
            else {
                --_i;
            }
            LBFGS_ASSERT(_obj != nullptr && _obj->is_valid_index(_i),
                         "iterator not decrementable");
            return *this;
        }

        constexpr auto operator--(int) noexcept -> type&
        {
            auto temp{*this};
            --(*this);
            return temp;
        }

        template <bool C>
        constexpr auto operator==(history_iterator<C> const& other) const
            noexcept -> bool
        {
            LBFGS_ASSERT(_obj == other._obj, "iterators pointing to different "
                                             "containers are not comparable");
            return _i == other._i;
        }

        template <bool C>
        constexpr auto operator!=(history_iterator<C> const& other) const
            noexcept -> bool
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

    constexpr auto sum(size_type const a, size_type const b) const noexcept
        -> size_type
    {
        auto r = a + b;
        r -= (r >= capacity()) * capacity();
        return r;
    }

    constexpr auto back_index() const noexcept -> size_type
    {
        return sum(_first, _size);
    }

    constexpr auto is_valid_index(size_type const i) const noexcept -> bool
    {
        auto r = (i < _first) ? (_first + size() >= capacity()
                                 && i < _first + size() - capacity())
                              : (i < std::min(_first + size(), capacity()));
        // LBFGS_TRACE(
        //     "is_valid_index(%zu)=%i, _first=%zu, size()=%zu, capacity()=%zu", i,
        //     r, _first, size(), capacity());
        return r;
    }

  public:
    constexpr auto begin() const noexcept -> const_iterator
    {
        return {this, _first};
    }
    constexpr auto begin() noexcept -> iterator { return {this, _first}; }

    constexpr auto end() const noexcept -> const_iterator
    {
        return {this, back_index()};
    }

    constexpr auto end() noexcept -> iterator { return {this, back_index()}; }

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
    float            value;
    float            dir_grad;
    gsl::span<float> x;
    gsl::span<float> grad;
};

struct lbfgs_state_t {
    iteration_history_t history;
    lbfgs_point_t       current;
    lbfgs_point_t       previous;
    gsl::span<float>    direction;
};

struct lbfgs_buffers_t {
    std::vector<float>            _workspace;
    std::vector<iteration_data_t> _history;
    size_t                        _n;

  private:
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
    lbfgs_buffers_t(size_t const n, size_t const m)
        : _workspace{}, _history{}, _n{}
    {
        _workspace.reserve(1048576UL);
        _history.reserve(32UL);
        resize(n, m);
    }

    auto resize(size_t n, size_t const m) -> void
    {
        if (n != _n || m != _history.size()) {
            // Since _workspace may need to be re-allocated, we don't want to
            // keep dangling pointers
            _history.clear();
            _history.resize(
                m, {0.0f, std::numeric_limits<float>::quiet_NaN(), {}, {}});
            _n = n;
            _workspace.resize(vector_size(n) * number_vectors(m));
            for (auto i = size_t{0}; i < _history.size(); ++i) {
                _history[i].s = get(2 * i);
                _history[i].y = get(2 * i + 1);
            }
        }
    }

    auto make_state() noexcept -> lbfgs_state_t
    {
        constexpr auto NaN = std::numeric_limits<float>::quiet_NaN();
        auto const     m   = _history.size();
        return lbfgs_state_t{{gsl::span<iteration_data_t>{_history}},
                             {NaN, NaN, get(m + 1), get(m + 2)},
                             {NaN, NaN, get(m + 3), get(m + 4)},
                             get(m + 5)};
    }
};

struct h0_t {
    double gamma;

    auto operator()(gsl::span<float> x) const noexcept -> void
    {
        detail::scal(static_cast<float>(gamma), x);
    }
};

template <class Function> struct line_search_func_t {
    ///< `float(gsl::span<float const>, gsl::span<float>)`
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
        "is "
        "also fine.");

    static constexpr auto is_noexcept() noexcept -> bool
    {
        return std::is_nothrow_invocable_r_v<
            double, Function, gsl::span<float const>, gsl::span<float>>;
    }

    auto operator()(float const alpha) const noexcept(is_noexcept())
    {
        for (auto i = size_t{0}; i < x.size(); ++i) {
            x[i] = x_0[i] + alpha * direction[i];
        }
        // Yes, we want implicit conversion to double here!
        double const f_x =
            value_and_gradient(static_cast<gsl::span<float const>>(x), grad);
        auto const df_x = detail::dot(grad, direction);
        return std::make_pair(f_x, df_x);
    }
};

template <class Iterator, class H0>
auto apply_inverse_hessian(Iterator begin, Iterator end, H0&& h0,
                           gsl::span<float> q)
{
    // for i = k − 1, k − 2, . . . , k − m
    std::for_each(std::make_reverse_iterator(end),
                  std::make_reverse_iterator(begin), [q](auto& x) {
                      // alpha_i <- rho_i*s_i^T*q
                      x.alpha = detail::dot(x.s, q) / x.s_dot_y;
                      // q <- q - alpha_i*y_i
                      detail::axpy(x.alpha, x.y, q);
                  });
    // r <- H_k^0*q
    h0(q);
    //for i = k − m, k − m + 1, . . . , k − 1
    std::for_each(begin, end, [q](auto& x) {
        // beta <- rho_i * y_i^T * r
        auto const beta = detail::dot(x.y, q) / x.s_dot_y;
        // r <- r + s_i * ( alpha_i - beta)
        detail::axpy(x.alpha - beta, x.s, q);
    });
    // stop with result "H_k*f_f'=q"
}

template <class Function>
void minimize(Function value_and_gradient, lbfgs_param_t const& params,
              lbfgs_state_t& state)
{
    state.current.value = value_and_gradient(
        gsl::span<float const>{state.current.x}, state.current.grad);

    // H₀ = 1
    h0_t h0{1};

    auto x_norm = detail::nrm2(state.current.x);
    auto g_norm = detail::nrm2(state.current.grad);
    if (g_norm / std::max(x_norm, 1.0) < params.epsilon) { return; }
    auto step = 1.0 / g_norm;

    std::transform(std::begin(state.current.grad), std::end(state.current.grad),
                   std::begin(state.direction),
                   [](auto const x) { return -x; });
    state.current.dir_grad = std::inner_product(
        std::begin(state.direction), std::end(state.direction),
        std::begin(state.current.grad), 0.0);
    LBFGS_TRACE("f(x_0) = %f, f'(x_0) = %f\n", state.current.value,
                state.current.dir_grad);

    for (;;) {
        state.previous.dir_grad = state.current.dir_grad;
        state.previous.value    = state.current.value;
        std::copy(std::begin(state.current.x), std::end(state.current.x),
                  std::begin(state.previous.x));
        std::copy(std::begin(state.current.grad), std::end(state.current.grad),
                  std::begin(state.previous.grad));

        auto fn = line_search_func_t<Function&>{
            value_and_gradient, state.current.x, state.current.grad,
            state.previous.x, state.direction};
        auto result = line_search(fn, params.line_search, state.previous.value,
                                  state.previous.dir_grad, step);
        if (result.status != status_t::success) { return; }
        if (!result.cached) { fn(result.step); }
        state.current.value = result.func;
        LBFGS_TRACE("--> f(x) = %f\n", state.current.value);
        for (auto const x : state.current.x) {
            std::printf("%f, ", x);
        }
        std::printf("\n");

        x_norm = detail::nrm2(state.current.x);
        g_norm = detail::nrm2(state.current.grad);
        if (g_norm / std::max(x_norm, 1.0) < params.epsilon) { return; }

        h0.gamma =
            state.history.emplace_back(state.current.x, state.previous.x,
                                       state.current.grad, state.previous.grad);

        std::transform(
            std::begin(state.current.grad), std::end(state.current.grad),
            std::begin(state.direction), [](auto const x) { return -x; });
        apply_inverse_hessian(state.history.begin(), state.history.end(), h0,
                              state.direction);

        for (auto const x : state.direction) {
            std::printf("%f, ", x);
        }
        std::printf("\n");

        step = 1.0;
    }
}

LBFGS_NAMESPACE_END

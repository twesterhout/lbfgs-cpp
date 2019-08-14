#include "line_search.hpp"
#include <catch2/catch.hpp>

TEST_CASE("Quadratic minimiser", "[quadratic]")
{
    auto const approx = [](auto const x) { return Approx(x).epsilon(1e-10); };
    // Using the parabola f(x) = 1/3 * (x - 11/2)^2 + 1/10
    REQUIRE(::LBFGS_NAMESPACE::detail::minimise_quadratic_interpolation(
                4.5, 0.4333333333, -0.6666666667, 7.5, 1.433333333)
            == approx(5.5));
    REQUIRE(::LBFGS_NAMESPACE::detail::minimise_quadratic_interpolation(
                6.5, 0.4333333333, 0.6666666666, 3.0, 2.183333333)
            == approx(5.5));
    // Using the parabola f(x) = 1/2 * (x + 1)^2 - 1/2
    REQUIRE(::LBFGS_NAMESPACE::detail::minimise_quadratic_interpolation(
                -5.0, -4.0, -3.0, -2.0)
            == approx(-1.0));
    REQUIRE(::LBFGS_NAMESPACE::detail::minimise_quadratic_interpolation(
                100.1, 101.1, 100.0, 101.0)
            == Approx(-1.0).epsilon(3e-4));
}

TEST_CASE("Cubic minimiser", "[cubic]")
{
    auto const approx = [](auto const x) { return Approx(x).epsilon(1e-9); };
    // Using f(x) = (x - 0.5) * (x - 3) * (x - 4) + 5
    // Extrema are at 1.459167 (min) and 3.540833 (max).
    REQUIRE(::LBFGS_NAMESPACE::detail::minimise_cubic_interpolation(
                0.0, 11.0, -15.5, 3.0, 5.0, 2.5)
            == approx(1.459167));
    REQUIRE(::LBFGS_NAMESPACE::detail::minimise_cubic_interpolation(
                0.1, 9.524, -14.03, 3.51, 5.752199, 0.1897)
            == approx(1.459167));
    REQUIRE(::LBFGS_NAMESPACE::detail::minimise_cubic_interpolation(
                1.456, 1.244893184, -0.019808, 3.54, 5.755136, 0.0052)
            == approx(1.459167));
    REQUIRE(::LBFGS_NAMESPACE::detail::minimise_cubic_interpolation(
                3.54, 5.755136, 0.0052, 1.456, 1.244893184, -0.019808)
            == approx(1.459167));
    REQUIRE(::LBFGS_NAMESPACE::detail::minimise_cubic_interpolation(
                2.0, 2.0, 2.5, -2.0, 80.0, -57.5)
            == approx(1.459167));
    REQUIRE(::LBFGS_NAMESPACE::detail::minimise_cubic_interpolation(
                0.0, 11.0, -15.5, 0.5, 5.0, -8.75)
            == approx(1.459167));
    REQUIRE(::LBFGS_NAMESPACE::detail::minimise_cubic_interpolation(
                0.5, 5.0, -8.75, 0.0, 11.0, -15.5)
            == approx(1.459167));
    REQUIRE(::LBFGS_NAMESPACE::detail::minimise_cubic_interpolation(
                -4.0, 257.0, -123.5, -4.1, 269.546, -127.43)
            == approx(1.459167));
}

#define TRY_WEIRD_PARAM(name, code, ...)                                       \
    for (auto const weird_##name : __VA_ARGS__) {                              \
        ::LBFGS_NAMESPACE::ls_param_t params;                                  \
        params.at_zero(1.0, -1.0);                                             \
        params.name = weird_##name;                                            \
        auto const [status, alpha, func, grad, num_f_evals, cache] =           \
            ::LBFGS_NAMESPACE::line_search(dummy_value_and_gradient, params,   \
                                           /*alpha_0=*/1.0);                   \
        REQUIRE(status == code);                                               \
        REQUIRE(alpha == 0.0);                                                 \
        REQUIRE((std::isnan(func) && std::isnan(params.func_0)                 \
                 || func == params.func_0));                                   \
        REQUIRE((std::isnan(grad) && std::isnan(params.grad_0)                 \
                 || grad == params.grad_0));                                   \
        REQUIRE(num_f_evals == 0);                                             \
        REQUIRE(cache == false);                                               \
    }                                                                          \
    do {                                                                       \
    } while (false)

TEST_CASE("Weird input", "[line_search]")
{
    auto       counter                  = size_t{0};
    auto const dummy_value_and_gradient = [&counter](auto /*unused*/) {
        ++counter;
        return std::make_pair(std::numeric_limits<double>::quiet_NaN(),
                              std::numeric_limits<double>::quiet_NaN());
    };
    TRY_WEIRD_PARAM(x_tol,
                    ::LBFGS_NAMESPACE::status_t::invalid_interval_tolerance,
                    {-1.0, -std::numeric_limits<double>::infinity(),
                     std::numeric_limits<double>::quiet_NaN()});
    TRY_WEIRD_PARAM(f_tol,
                    ::LBFGS_NAMESPACE::status_t::invalid_function_tolerance,
                    {-1e-10, -1.0, -std::numeric_limits<double>::infinity(),
                     std::numeric_limits<double>::quiet_NaN()});
    TRY_WEIRD_PARAM(g_tol,
                    ::LBFGS_NAMESPACE::status_t::invalid_gradient_tolerance,
                    {-1e-10, -1.0, -std::numeric_limits<double>::infinity(),
                     std::numeric_limits<double>::quiet_NaN()});
    TRY_WEIRD_PARAM(step_min, ::LBFGS_NAMESPACE::status_t::invalid_step_bounds,
                    {-1e-10, 0.0, 1e100,
                     std::numeric_limits<double>::infinity(),
                     std::numeric_limits<double>::quiet_NaN()});
    TRY_WEIRD_PARAM(
        step_max, ::LBFGS_NAMESPACE::status_t::invalid_step_bounds,
        {-1e-10, 0.0, 1e-100, std::numeric_limits<double>::quiet_NaN()});
    TRY_WEIRD_PARAM(func_0, ::LBFGS_NAMESPACE::status_t::invalid_argument,
                    {std::numeric_limits<double>::quiet_NaN()});
    TRY_WEIRD_PARAM(grad_0, ::LBFGS_NAMESPACE::status_t::invalid_argument,
                    {std::numeric_limits<double>::quiet_NaN(), 1.0,
                     std::numeric_limits<double>::infinity()});

    // Try negative initial step size
    {
        ::LBFGS_NAMESPACE::ls_param_t params;
        auto const [status, alpha, func, grad, num_f_evals, cache] =
            ::LBFGS_NAMESPACE::line_search(dummy_value_and_gradient,
                                           params.at_zero(1.0, -1.0),
                                           /*alpha_0=*/-0.5);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::invalid_argument);
        REQUIRE(alpha == 0.0);
        REQUIRE((std::isnan(func) && std::isnan(params.func_0)
                 || func == params.func_0));
        REQUIRE((std::isnan(grad) && std::isnan(params.grad_0)
                 || grad == params.grad_0));
        REQUIRE(num_f_evals == 0);
        REQUIRE(cache == false);
    }
}

#undef TRY_WEIRD_PARAM

TEST_CASE("Function (5.1)", "[line_search]")
{
    auto const value_and_gradient = [](auto const x) {
        static_assert(std::is_same_v<std::remove_const_t<decltype(x)>, double>);
        constexpr auto beta = 2.0;
        auto const     temp = x * x + beta;
        auto const     func = -x / temp;
        auto const     grad = (x * x - beta) / (temp * temp);
        return std::make_pair(func, grad);
    };
    auto const [func_0, grad_0] = value_and_gradient(0.0);

    {
        ::LBFGS_NAMESPACE::ls_param_t params;
        params.f_tol = 1e-3;
        params.g_tol = 1e-1;
        auto const [status, alpha, _1, grad, num_f_evals, _2] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient,
                                           params.at_zero(func_0, grad_0),
                                           /*alpha_0=*/0.001);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        REQUIRE(alpha == Approx(1.4).epsilon(5e-2));
        REQUIRE(grad == Approx(-9.2e-3).epsilon(1e-2));
        REQUIRE(num_f_evals == 6);
    }

    {
        ::LBFGS_NAMESPACE::ls_param_t params;
        params.f_tol = 1e-3;
        params.g_tol = 1e-1;
        auto const [status, alpha, _1, grad, num_f_evals, _2] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient,
                                           params.at_zero(func_0, grad_0),
                                           /*alpha_0=*/0.1);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        REQUIRE(alpha == Approx(1.4).epsilon(1e-1));
        REQUIRE(grad == Approx(4.7e-3).epsilon(1e-1));
        REQUIRE(num_f_evals == 3);
    }

    {
        ::LBFGS_NAMESPACE::ls_param_t params;
        params.f_tol = 1e-3;
        params.g_tol = 1e-1;
        auto const [status, alpha, _1, grad, num_f_evals, _2] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient,
                                           params.at_zero(func_0, grad_0),
                                           /*alpha_0=*/10.0);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        REQUIRE(alpha == Approx(10).epsilon(1e-2));
        REQUIRE(grad == Approx(9.4e-3).epsilon(1e-2));
        REQUIRE(num_f_evals == 1);
    }

    {
        ::LBFGS_NAMESPACE::ls_param_t params;
        params.f_tol = 1e-3;
        params.g_tol = 1e-1;
        auto const [status, alpha, _1, grad, num_f_evals, _2] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient,
                                           params.at_zero(func_0, grad_0),
                                           /*alpha_0=*/1000.0);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        REQUIRE(alpha == Approx(37).epsilon(1e-2));
        REQUIRE(grad == Approx(7.3e-4).epsilon(1e-2));
        REQUIRE(num_f_evals == 4);
    }

    {
        ::LBFGS_NAMESPACE::ls_param_t params;
        params.f_tol = 1e-1;
        params.g_tol = 1e-1;
        auto const [status, alpha, _1, grad, num_f_evals, _2] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient,
                                           params.at_zero(func_0, grad_0),
                                           /*alpha_0=*/10.0);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        REQUIRE(alpha == Approx(1.6).epsilon(4e-2));
    }

    {
        ::LBFGS_NAMESPACE::ls_param_t params;
        params.f_tol = 1e-1;
        params.g_tol = 1e-1;
        auto const [status, alpha, _1, grad, num_f_evals, _2] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient,
                                           params.at_zero(func_0, grad_0),
                                           /*alpha_0=*/1000.0);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        REQUIRE(alpha == Approx(1.6).epsilon(2e-2));
    }

    {
        ::LBFGS_NAMESPACE::ls_param_t params;
        params.f_tol = 1e-1;
        params.g_tol = 1e-4;
        auto const [status, alpha, _1, grad, num_f_evals, _2] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient,
                                           params.at_zero(func_0, grad_0),
                                           /*alpha_0=*/0.001);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        REQUIRE(alpha == Approx(1.414213562).epsilon(1e-4));
        REQUIRE(num_f_evals == 8);
    }

    {
        ::LBFGS_NAMESPACE::ls_param_t params;
        params.f_tol = 1e-1;
        params.g_tol = 1e-4;
        auto const [status, alpha, _1, grad, num_f_evals, _2] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient,
                                           params.at_zero(func_0, grad_0),
                                           /*alpha_0=*/0.1);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        REQUIRE(alpha == Approx(1.414213562).epsilon(1e-4));
        // NOTE: In the paper, they get 6 here. But it's okay since 5 < 6.
        REQUIRE(num_f_evals == 5);
    }

    {
        ::LBFGS_NAMESPACE::ls_param_t params;
        params.f_tol = 1e-1;
        params.g_tol = 1e-4;
        auto const [status, alpha, _1, grad, num_f_evals, _2] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient,
                                           params.at_zero(func_0, grad_0),
                                           /*alpha_0=*/10.0);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        REQUIRE(alpha == Approx(1.414213562).epsilon(1e-3));
        REQUIRE(num_f_evals == 6);
    }

    {
        ::LBFGS_NAMESPACE::ls_param_t params;
        params.f_tol = 1e-1;
        params.g_tol = 1e-4;
        auto const [status, alpha, _1, grad, num_f_evals, _2] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient,
                                           params.at_zero(func_0, grad_0),
                                           /*alpha_0=*/1000.0);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        REQUIRE(alpha == Approx(1.414213562).epsilon(1e-4));
        REQUIRE(num_f_evals == 10);
    }

    // Small interval
    {
        ::LBFGS_NAMESPACE::ls_param_t params;
        params.f_tol = 1e-3;
        params.g_tol = 1e-1;
        params.x_tol = 1.0; // !!!
        auto const [status, alpha, _1, grad, num_f_evals, _2] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient,
                                           params.at_zero(func_0, grad_0),
                                           /*alpha_0=*/1000.0);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::interval_too_small);
    }

    // Require too big a step
    {
        ::LBFGS_NAMESPACE::ls_param_t params;
        params.f_tol    = 1e-3;
        params.g_tol    = 1e-1;
        params.step_min = 2.0; // !!!
        auto const [status, alpha, _1, grad, num_f_evals, _2] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient,
                                           params.at_zero(func_0, grad_0),
                                           /*alpha_0=*/0.001);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::minimum_step_reached);
    }

    // Require too small a step
    {
        ::LBFGS_NAMESPACE::ls_param_t params;
        params.f_tol    = 1e-3;
        params.g_tol    = 1e-1;
        params.step_max = 0.5; // !!!
        auto const [status, alpha, _1, grad, num_f_evals, _2] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient,
                                           params.at_zero(func_0, grad_0),
                                           /*alpha_0=*/0.001);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::maximum_step_reached);
    }
}

TEST_CASE("Function (5.2)", "[line_search]")
{
    auto const value_and_gradient = [](auto const x) {
        static_assert(std::is_same_v<std::remove_const_t<decltype(x)>, double>);
        constexpr auto beta = 0.004;
        auto const     func =
            std::pow(x + beta, 5.0) - 2.0 * std::pow(x + beta, 4.0);
        auto const grad =
            5.0 * std::pow(x + beta, 4.0) - 8.0 * std::pow(x + beta, 3.0);
        return std::make_pair(func, grad);
    };
    auto const [func_0, grad_0] = value_and_gradient(0.0);

    {
        ::LBFGS_NAMESPACE::ls_param_t params;
        params.f_tol = 1e-1;
        params.g_tol = 1e-1;
        auto const [status, alpha, _1, grad, num_f_evals, _2] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient,
                                           params.at_zero(func_0, grad_0),
                                           /*alpha_0=*/0.001);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        REQUIRE(alpha == Approx(1.596).epsilon(1e-2));
        REQUIRE(grad == Approx(7.1e-9).margin(1e-8));
        REQUIRE(num_f_evals == 12);
    }

    {
        ::LBFGS_NAMESPACE::ls_param_t params;
        params.f_tol = 1e-1;
        params.g_tol = 1e-1;
        auto const [status, alpha, _1, grad, num_f_evals, _2] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient,
                                           params.at_zero(func_0, grad_0),
                                           /*alpha_0=*/0.1);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        REQUIRE(alpha == Approx(1.596).epsilon(1e-2));
        REQUIRE(grad == Approx(10e-10).margin(1e-9));
        REQUIRE(num_f_evals == 8);
    }

    {
        ::LBFGS_NAMESPACE::ls_param_t params;
        params.f_tol = 1e-1;
        params.g_tol = 1e-1;
        auto const [status, alpha, _1, grad, num_f_evals, _2] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient,
                                           params.at_zero(func_0, grad_0),
                                           /*alpha_0=*/10.0);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        REQUIRE(alpha == Approx(1.596).epsilon(1e-2));
        REQUIRE(grad == Approx(-5e-9).margin(1e-9));
        REQUIRE(num_f_evals == 8);
    }

    {
        ::LBFGS_NAMESPACE::ls_param_t params;
        params.f_tol = 1e-1;
        params.g_tol = 1e-1;
        auto const [status, alpha, _1, grad, num_f_evals, _2] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient,
                                           params.at_zero(func_0, grad_0),
                                           /*alpha_0=*/1000.0);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        REQUIRE(alpha == Approx(1.596).epsilon(1e-2));
        REQUIRE(grad == Approx(-2.3e-8).margin(1e-9));
        REQUIRE(num_f_evals == 11);
    }
}

TEST_CASE("Function (5.3)", "[line_search]")
{
    auto const value_and_gradient = [](auto const x) {
        static_assert(std::is_same_v<std::remove_const_t<decltype(x)>, double>);
        constexpr auto l    = 39.0;
        constexpr auto beta = 0.01;
        constexpr auto pi   = M_PI;
        auto const     f_0  = [beta](auto const _x) {
            if (_x <= 1.0 - beta) return 1.0 - _x;
            if (_x >= 1.0 + beta) return _x - 1.0;
            return (_x - 1.0) * (_x - 1.0) / (2.0 * beta) + beta / 2;
        };
        auto const g_0 = [beta](auto const _x) {
            if (_x <= 1.0 - beta) return -1.0;
            if (_x >= 1.0 + beta) return 1.0;
            return (_x - 1.0) / beta;
        };
        auto const func =
            f_0(x) + 2.0 * (1.0 - beta) / (l * pi) * std::sin(l * pi / 2.0 * x);
        auto const grad = g_0(x) + (1.0 - beta) * std::cos(l * pi / 2.0 * x);
        return std::make_pair(func, grad);
    };
    auto const [func_0, grad_0] = value_and_gradient(0.0);

    {
        ::LBFGS_NAMESPACE::ls_param_t params;
        params.f_tol = 1e-1;
        params.g_tol = 1e-1;
        auto const [status, alpha, _1, grad, num_f_evals, _2] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient,
                                           params.at_zero(func_0, grad_0),
                                           /*alpha_0=*/0.001);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        REQUIRE(alpha == Approx(1.0).epsilon(1e-2));
        REQUIRE(grad == Approx(-5.1e-5).margin(2e-5));
        REQUIRE(num_f_evals == 12);
    }

    {
        ::LBFGS_NAMESPACE::ls_param_t params;
        params.f_tol = 1e-1;
        params.g_tol = 1e-1;
        auto const [status, alpha, _1, grad, num_f_evals, _2] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient,
                                           params.at_zero(func_0, grad_0),
                                           /*alpha_0=*/0.1);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        REQUIRE(alpha == Approx(1.0).epsilon(1e-2));
        REQUIRE(grad == Approx(-1.9e-4).margin(2e-5));
        REQUIRE(num_f_evals == 12);
    }

    {
        ::LBFGS_NAMESPACE::ls_param_t params;
        params.f_tol = 1e-1;
        params.g_tol = 1e-1;
        auto const [status, alpha, _1, grad, num_f_evals, _2] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient,
                                           params.at_zero(func_0, grad_0),
                                           /*alpha_0=*/10.0);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        REQUIRE(alpha == Approx(1.0).epsilon(1e-2));
        REQUIRE(grad == Approx(-2.0e-6).margin(1e-5));
        REQUIRE(num_f_evals == 10);
    }

    {
        ::LBFGS_NAMESPACE::ls_param_t params;
        params.f_tol = 1e-1;
        params.g_tol = 1e-1;
        auto const [status, alpha, _1, grad, num_f_evals, _2] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient,
                                           params.at_zero(func_0, grad_0),
                                           /*alpha_0=*/1000.0);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        REQUIRE(alpha == Approx(1.0).epsilon(1e-2));
        REQUIRE(grad == Approx(-1.6e-5).margin(5e-5));
        REQUIRE(num_f_evals == 13);
    }
}

TEST_CASE("Function (5.4) β₁=0.001, β₂=0.001", "[line_search]")
{
    auto const value_and_gradient = [](auto const x) {
        static_assert(std::is_same_v<std::remove_const_t<decltype(x)>, double>);
        auto const gamma = [](auto const beta) {
            return std::sqrt(1.0 + beta * beta) - beta;
        };
        constexpr auto beta_1  = 0.001;
        constexpr auto beta_2  = 0.001;
        constexpr auto gamma_1 = 0.9990005;
        constexpr auto gamma_2 = 0.9990005;
        auto const     a = std::sqrt((1.0 - x) * (1.0 - x) + beta_2 * beta_2);
        auto const     b = std::sqrt(x * x + beta_1 * beta_1);
        auto const     func = gamma_1 * a + gamma_2 * b;
        auto const     grad = gamma_2 * x / b + gamma_1 * (x - 1.0) / a;
        return std::make_pair(func, grad);
    };
    auto const [func_0, grad_0] = value_and_gradient(0.0);
    REQUIRE(grad_0 == Approx(-0.9990000005).epsilon(1e-5));
    REQUIRE(func_0 == Approx(1.0).epsilon(1e-5));

    {
        ::LBFGS_NAMESPACE::ls_param_t params;
        params.f_tol = 1e-3;
        params.g_tol = 1e-3;
        auto const [status, alpha, _1, grad, num_f_evals, _2] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient,
                                           params.at_zero(func_0, grad_0),
                                           /*alpha_0=*/0.001);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        // REQUIRE(alpha == Approx(0.08f).epsilon(1e-2f));
        // REQUIRE(grad == Approx(-6.9e-5f).margin(1e-5f));
        // REQUIRE(num_f_evals == 4);
    }

    {
        ::LBFGS_NAMESPACE::ls_param_t params;
        params.f_tol = 1e-3;
        params.g_tol = 1e-3;
        auto const [status, alpha, _1, grad, num_f_evals, _2] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient,
                                           params.at_zero(func_0, grad_0),
                                           /*alpha_0=*/0.1);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        // REQUIRE(alpha == Approx(0.08f).epsilon(1e-2f));
        // REQUIRE(grad == Approx(-6.9e-5f).margin(1e-5f));
        // REQUIRE(num_f_evals == 4);
    }

    {
        ::LBFGS_NAMESPACE::ls_param_t params;
        params.f_tol = 1e-3;
        params.g_tol = 1e-3;
        auto const [status, alpha, _1, grad, num_f_evals, _2] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient,
                                           params.at_zero(func_0, grad_0),
                                           /*alpha_0=*/10.0);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        // REQUIRE(alpha == Approx(0.08f).epsilon(1e-2f));
        // REQUIRE(grad == Approx(-6.9e-5f).margin(1e-5f));
        // REQUIRE(num_f_evals == 4);
    }

    {
        ::LBFGS_NAMESPACE::ls_param_t params;
        params.f_tol = 1e-3;
        params.g_tol = 1e-3;
        auto const [status, alpha, _1, grad, num_f_evals, _2] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient,
                                           params.at_zero(func_0, grad_0),
                                           /*alpha_0=*/1000.0);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        // REQUIRE(alpha == Approx(0.08f).epsilon(1e-2f));
        // REQUIRE(grad == Approx(-6.9e-5f).margin(1e-5f));
        // REQUIRE(num_f_evals == 4);
    }
}

TEST_CASE("Function (5.4) β₁=0.01, β₂=0.001", "[line_search]")
{
    auto const value_and_gradient = [](auto const x) {
        static_assert(std::is_same_v<std::remove_const_t<decltype(x)>, double>);
        auto const gamma = [](auto const beta) {
            return std::sqrt(1.0 + beta * beta) - beta;
        };
        constexpr auto beta_1  = 0.01;
        constexpr auto beta_2  = 0.001;
        constexpr auto gamma_1 = 0.9900499990;
        constexpr auto gamma_2 = 0.9990005;
        auto const     a = std::sqrt((1.0 - x) * (1.0 - x) + beta_2 * beta_2);
        auto const     b = std::sqrt(x * x + beta_1 * beta_1);
        auto const     func = gamma_1 * a + gamma_2 * b;
        auto const     grad = gamma_2 * x / b + gamma_1 * (x - 1.0) / a;
        return std::make_pair(func, grad);
    };
    auto const [func_0, grad_0] = value_and_gradient(0.0);
    REQUIRE(grad_0 == Approx(-0.9900495039).epsilon(1e-5));
    REQUIRE(func_0 == Approx(1.000040499).epsilon(1e-5));

    {
        ::LBFGS_NAMESPACE::ls_param_t params;
        params.f_tol = 1e-3;
        params.g_tol = 1e-3;
        auto const [status, alpha, _1, grad, num_f_evals, _2] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient,
                                           params.at_zero(func_0, grad_0),
                                           /*alpha_0=*/0.001);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        REQUIRE(alpha == Approx(0.075).epsilon(1e-2));
        REQUIRE(grad == Approx(1.9e-4).margin(5e-5));
        // REQUIRE(num_f_evals == 6);
    }

    {
        ::LBFGS_NAMESPACE::ls_param_t params;
        params.f_tol = 1e-3;
        params.g_tol = 1e-3;
        auto const [status, alpha, _1, grad, num_f_evals, _2] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient,
                                           params.at_zero(func_0, grad_0),
                                           /*alpha_0=*/0.1);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        REQUIRE(alpha == Approx(0.078).epsilon(1e-2));
        REQUIRE(grad == Approx(7.4e-4).margin(2e-5));
        REQUIRE(num_f_evals == 3);
    }

    {
        ::LBFGS_NAMESPACE::ls_param_t params;
        params.f_tol = 1e-3;
        params.g_tol = 1e-3;
        auto const [status, alpha, _1, grad, num_f_evals, _2] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient,
                                           params.at_zero(func_0, grad_0),
                                           /*alpha_0=*/10.0);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        REQUIRE(alpha == Approx(0.073).epsilon(1e-2));
        REQUIRE(grad == Approx(-2.6e-4).margin(2e-5));
        REQUIRE(num_f_evals == 7);
    }

    {
        ::LBFGS_NAMESPACE::ls_param_t params;
        params.f_tol = 1e-3;
        params.g_tol = 1e-3;
        auto const [status, alpha, _1, grad, num_f_evals, _2] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient,
                                           params.at_zero(func_0, grad_0),
                                           /*alpha_0=*/1000.0);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        REQUIRE(alpha == Approx(0.076).epsilon(1e-2));
        REQUIRE(grad == Approx(4.5e-4).margin(2e-5));
        REQUIRE(num_f_evals == 8);
    }
}

TEST_CASE("Function (5.4) β₁=0.001, β₂=0.01", "[line_search]")
{
    auto const value_and_gradient = [](auto const x) {
        static_assert(std::is_same_v<std::remove_const_t<decltype(x)>, double>);
        auto const gamma = [](auto const beta) {
            return std::sqrt(1.0f + beta * beta) - beta;
        };
        constexpr auto beta_1  = 0.001;
        constexpr auto beta_2  = 0.01;
        constexpr auto gamma_1 = 0.9990005;
        constexpr auto gamma_2 = 0.9900499990;
        auto const     a = std::sqrt((1.0 - x) * (1.0 - x) + beta_2 * beta_2);
        auto const     b = std::sqrt(x * x + beta_1 * beta_1);
        auto const     func = gamma_1 * a + gamma_2 * b;
        auto const     grad = gamma_2 * x / b + gamma_1 * (x - 1.0) / a;
        return std::make_pair(func, grad);
    };
    auto const [func_0, grad_0] = value_and_gradient(0.0);
    REQUIRE(grad_0 == Approx(-0.9989505539).epsilon(1e-5));
    REQUIRE(func_0 == Approx(1.000040499).epsilon(1e-5));

    {
        ::LBFGS_NAMESPACE::ls_param_t params;
        params.f_tol = 1e-3;
        params.g_tol = 1e-3;
        auto const [status, alpha, _1, grad, num_f_evals, _2] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient,
                                           params.at_zero(func_0, grad_0),
                                           /*alpha_0=*/0.001);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        REQUIRE(alpha == Approx(0.93).epsilon(1e-2));
        REQUIRE(grad == Approx(5.2e-4).margin(2e-3));
        // REQUIRE(num_f_evals == 13);
    }

    {
        ::LBFGS_NAMESPACE::ls_param_t params;
        params.f_tol = 1e-3;
        params.g_tol = 1e-3;
        auto const [status, alpha, _1, grad, num_f_evals, _2] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient,
                                           params.at_zero(func_0, grad_0),
                                           /*alpha_0=*/0.1);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        REQUIRE(alpha == Approx(0.93).epsilon(1e-2));
        REQUIRE(grad == Approx(8.4e-5).margin(5e-4));
        // REQUIRE(num_f_evals == 11);
    }

    {
        ::LBFGS_NAMESPACE::ls_param_t params;
        params.f_tol = 1e-3;
        params.g_tol = 1e-3;
        auto const [status, alpha, _1, grad, num_f_evals, _2] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient,
                                           params.at_zero(func_0, grad_0),
                                           /*alpha_0=*/10.0);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        REQUIRE(alpha == Approx(0.92).epsilon(1e-2));
        REQUIRE(grad == Approx(-2.4e-4).margin(2e-4));
        // REQUIRE(num_f_evals == 8);
    }

    {
        ::LBFGS_NAMESPACE::ls_param_t params;
        params.f_tol = 1e-3;
        params.g_tol = 1e-3;
        auto const [status, alpha, _1, grad, num_f_evals, _2] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient,
                                           params.at_zero(func_0, grad_0),
                                           /*alpha_0=*/1000.0);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        REQUIRE(alpha == Approx(0.92).epsilon(1e-2));
        REQUIRE(grad == Approx(-3.2e-4).margin(2e-4));
        // REQUIRE(num_f_evals == 11);
    }
}

TEST_CASE("Quadratic Function", "[LineSearches.jl:alphacalc]")
{
    {
        auto const value_and_gradient = [](auto const alpha) {
            auto const t  = -1.0f + 2.0f * static_cast<float>(alpha);
            auto const f  = 2.0f * t * t;
            auto const df = 4.0f * t;
            return std::make_pair(f, df);
        };
        auto const [func_0, grad_0] = value_and_gradient(0.0f);
#if defined(LBFGS_CLANG)
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wdouble-promotion"
#endif
        auto const [status, alpha, _1, grad, num_f_evals, _2] =
            ::LBFGS_NAMESPACE::line_search(
                value_and_gradient,
                ::LBFGS_NAMESPACE::ls_param_t{}.at_zero(func_0, grad_0),
                /*alpha_0=*/1.0);
#if defined(LBFGS_CLANG)
#    pragma clang diagnostic pop
#endif
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        REQUIRE(alpha == Approx(0.49995));
    }

    {
        auto const value_and_gradient = [](auto const alpha) {
            auto const t  = 2.0f * static_cast<float>(alpha);
            auto const f  = 2.0f * t * t;
            auto const df = 4.0f * t;
            return std::make_pair(f, df);
        };
        auto const [func_0, grad_0] = value_and_gradient(0.0f);
#if defined(LBFGS_CLANG)
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wdouble-promotion"
#endif
        auto const [status, alpha, _1, grad, num_f_evals, _2] =
            ::LBFGS_NAMESPACE::line_search(
                value_and_gradient,
                ::LBFGS_NAMESPACE::ls_param_t{}.at_zero(func_0, grad_0),
                /*alpha_0=*/1.0);
#if defined(LBFGS_CLANG)
#    pragma clang diagnostic pop
#endif
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::invalid_argument);
        REQUIRE(alpha == Approx(0.0));
    }
}

TEST_CASE("Himmelblau's Function", "[LineSearches.jl:alphacalc]")
{
    auto const himmelblau = [](std::array<float, 2> const x) {
        auto const t1 = x[0] * x[0] + x[1] - 11.0f;
        auto const t2 = x[0] + x[1] * x[1] - 7.0f;
        return t1 * t1 + t2 * t2;
    };

    auto const himmelblau_gradient = [](std::array<float, 2> const x) {
        return std::array<float, 2>{
            4.0f * x[0] * x[0] * x[0] + 4.0f * x[0] * x[1] - 44.0f * x[0]
                + 2.0f * x[0] + 2.0f * x[1] * x[1] - 14.0f,
            2.0f * x[0] * x[0] + 2.0f * x[1] - 22.0f + 4.0f * x[0] * x[1]
                + 4.0f * x[1] * x[1] * x[1] - 28.0f * x[1]};
    };

    auto const x0 = std::array<float, 2>{2.0f, 2.0f};
    auto const s  = std::array<float, 2>{42.0f, 18.0f};

    auto const value_and_gradient = [=](auto const alpha) {
        auto const x =
            std::array<float, 2>{x0[0] + static_cast<float>(alpha) * s[0],
                                 x0[1] + static_cast<float>(alpha) * s[1]};
        auto const f  = himmelblau(x);
        auto const df = himmelblau_gradient(x);
        return std::make_pair(f, df[0] * s[0] + df[1] * s[1]);
    };
    auto const [func_0, grad_0] = value_and_gradient(0.0f);
    REQUIRE(func_0 == Approx(26.0));
    REQUIRE(grad_0 == Approx(-2088.0));

#if defined(LBFGS_CLANG)
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wdouble-promotion"
#endif
    auto const [status, alpha, _1, grad, num_f_evals, _2] =
        ::LBFGS_NAMESPACE::line_search(
            value_and_gradient,
            ::LBFGS_NAMESPACE::ls_param_t{}.at_zero(func_0, grad_0),
            /*alpha_0=*/1.0);
#if defined(LBFGS_CLANG)
#    pragma clang diagnostic pop
#endif
    REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
    REQUIRE(alpha == Approx(0.0175892025844326));
}

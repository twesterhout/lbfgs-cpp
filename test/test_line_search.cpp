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
            == Approx(-1.0).epsilon(3e-3));
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

TEST_CASE("Function (5.1)", "[line_search]")
{
    auto const value_and_gradient = [](auto const x) {
        static_assert(std::is_same_v<std::remove_const_t<decltype(x)>, float>);
        constexpr auto beta = 2.0f;
        auto const     temp = x * x + beta;
        auto const     func = -x / temp;
        auto const     grad = (x * x - beta) / (temp * temp);
        return std::make_pair(func, grad);
    };
    auto const [func_0, grad_0] = value_and_gradient(0.0f);

    {
        ::LBFGS_NAMESPACE::param_type params;
        params.f_tol = 1e-3f;
        params.g_tol = 1e-1f;
        auto const [status, alpha, _, grad, num_f_evals] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient, params, func_0,
                                           grad_0, /*alpha_0=*/0.001f);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        REQUIRE(alpha == Approx(1.4f).epsilon(5e-2f));
        REQUIRE(grad == Approx(-9.2e-3f).epsilon(1e-1f));
        REQUIRE(num_f_evals == 6);
    }

    {
        ::LBFGS_NAMESPACE::param_type params;
        params.f_tol = 1e-3f;
        params.g_tol = 1e-1f;
        auto const [status, alpha, _, grad, num_f_evals] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient, params, func_0,
                                           grad_0, /*alpha_0=*/0.1f);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        REQUIRE(alpha == Approx(1.4).epsilon(1e-1));
        REQUIRE(grad == Approx(4.7e-3f).epsilon(1e-1f));
        REQUIRE(num_f_evals == 3);
    }

    {
        ::LBFGS_NAMESPACE::param_type params;
        params.f_tol = 1e-3f;
        params.g_tol = 1e-1f;
        auto const [status, alpha, _, grad, num_f_evals] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient, params, func_0,
                                           grad_0, /*alpha_0=*/10.0f);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        REQUIRE(alpha == Approx(10).epsilon(1e-1));
        REQUIRE(grad == Approx(9.4e-3f).epsilon(1e-1f));
        REQUIRE(num_f_evals == 1);
    }

    {
        ::LBFGS_NAMESPACE::param_type params;
        params.f_tol = 1e-3f;
        params.g_tol = 1e-1f;
        auto const [status, alpha, _, grad, num_f_evals] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient, params, func_0,
                                           grad_0, /*alpha_0=*/1000.0f);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        REQUIRE(alpha == Approx(37).epsilon(1e-1));
        REQUIRE(grad == Approx(7.3e-4f).epsilon(1e-1f));
        REQUIRE(num_f_evals == 4);
    }

    {
        ::LBFGS_NAMESPACE::param_type params;
        params.f_tol = 1e-1f;
        params.g_tol = 1e-1f;
        auto const [status, alpha, _, grad, num_f_evals] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient, params, func_0,
                                           grad_0, /*alpha_0=*/10.0f);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        REQUIRE(alpha == Approx(1.6).epsilon(4e-2));
    }

    {
        ::LBFGS_NAMESPACE::param_type params;
        params.f_tol = 1e-1f;
        params.g_tol = 1e-1f;
        auto const [status, alpha, _, grad, num_f_evals] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient, params, func_0,
                                           grad_0, /*alpha_0=*/1000.0f);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        REQUIRE(alpha == Approx(1.6).epsilon(2e-2));
    }

    {
        ::LBFGS_NAMESPACE::param_type params;
        params.f_tol = 1e-1f;
        params.g_tol = 1e-4f;
        auto const [status, alpha, _, grad, num_f_evals] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient, params, func_0,
                                           grad_0, /*alpha_0=*/0.001f);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        REQUIRE(alpha == Approx(1.414213562f).epsilon(1e-4));
        REQUIRE(num_f_evals == 8);
    }

    {
        ::LBFGS_NAMESPACE::param_type params;
        params.f_tol = 1e-1f;
        params.g_tol = 1e-4f;
        auto const [status, alpha, _, grad, num_f_evals] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient, params, func_0,
                                           grad_0, /*alpha_0=*/0.1f);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        REQUIRE(alpha == Approx(1.414213562f).epsilon(1e-4));
        // NOTE: In the paper, they get 6 here. But it's okay since 5 < 6.
        REQUIRE(num_f_evals == 5);
    }

    {
        ::LBFGS_NAMESPACE::param_type params;
        params.f_tol = 1e-1f;
        params.g_tol = 1e-4f;
        auto const [status, alpha, _, grad, num_f_evals] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient, params, func_0,
                                           grad_0, /*alpha_0=*/10.0f);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        REQUIRE(alpha == Approx(1.414213562f).epsilon(1e-3));
        REQUIRE(num_f_evals == 6);
    }

    {
        ::LBFGS_NAMESPACE::param_type params;
        params.f_tol = 1e-1f;
        params.g_tol = 1e-4f;
        auto const [status, alpha, _, grad, num_f_evals] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient, params, func_0,
                                           grad_0, /*alpha_0=*/1000.0f);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        REQUIRE(alpha == Approx(1.414213562f).epsilon(1e-4));
        REQUIRE(num_f_evals == 10);
    }
}

TEST_CASE("Function (5.2)", "[line_search]")
{
    auto const value_and_gradient = [](auto const x) {
        static_assert(std::is_same_v<std::remove_const_t<decltype(x)>, float>);
        constexpr auto beta = 0.004f;
        auto const     func =
            std::pow(x + beta, 5.0f) - 2.0f * std::pow(x + beta, 4.0f);
        auto const grad =
            5.0f * std::pow(x + beta, 4.0f) - 8.0f * std::pow(x + beta, 3.0f);
        return std::make_pair(func, grad);
    };
    auto const [func_0, grad_0] = value_and_gradient(0.0f);

    {
        ::LBFGS_NAMESPACE::param_type params;
        params.f_tol = 1e-1f;
        params.g_tol = 1e-1f;
        auto const [status, alpha, _, grad, num_f_evals] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient, params, func_0,
                                           grad_0, /*alpha_0=*/0.001f);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        REQUIRE(alpha == Approx(1.596f).epsilon(1e-2f));
        REQUIRE(grad == Approx(7.1e-9f).margin(1e-8f));
        // NOTE: In the paper, they claim that they obtain the result within 12
        // function evaluations. Implementation from CppNumericalSolvers does as
        // many as 16 function evaluations.
        REQUIRE(num_f_evals == 15);
    }

    {
        ::LBFGS_NAMESPACE::param_type params;
        params.f_tol = 1e-1f;
        params.g_tol = 1e-1f;
        auto const [status, alpha, _, grad, num_f_evals] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient, params, func_0,
                                           grad_0, /*alpha_0=*/0.1f);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        REQUIRE(alpha == Approx(1.596f).epsilon(1e-2f));
        REQUIRE(grad == Approx(0).margin(1e-9f));
        REQUIRE(num_f_evals == 8);
    }

    {
        ::LBFGS_NAMESPACE::param_type params;
        params.f_tol = 1e-1f;
        params.g_tol = 1e-1f;
        auto const [status, alpha, _, grad, num_f_evals] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient, params, func_0,
                                           grad_0, /*alpha_0=*/10.0f);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        REQUIRE(alpha == Approx(1.596f).epsilon(1e-2f));
        REQUIRE(grad == Approx(0).margin(1e-9f));
        REQUIRE(num_f_evals == 8);
    }

    {
        ::LBFGS_NAMESPACE::param_type params;
        params.f_tol = 1e-1f;
        params.g_tol = 1e-1f;
        auto const [status, alpha, _, grad, num_f_evals] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient, params, func_0,
                                           grad_0, /*alpha_0=*/1000.0f);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        REQUIRE(alpha == Approx(1.596f).epsilon(1e-2f));
        REQUIRE(grad == Approx(0).margin(1e-9f));
        REQUIRE(num_f_evals == 11);
    }
}

TEST_CASE("Function (5.3)", "[line_search]")
{
    auto const value_and_gradient = [](auto const x) {
        static_assert(std::is_same_v<std::remove_const_t<decltype(x)>, float>);
        constexpr auto l    = 39.0f;
        constexpr auto beta = 0.01f;
        constexpr auto pi   = static_cast<float>(M_PI);
        auto const     f_0  = [beta](auto const _x) {
            if (_x <= 1.0f - beta) return 1.0f - _x;
            if (_x >= 1.0f + beta) return _x - 1.0f;
            return (_x - 1.0f) * (_x - 1.0f) / (2.0f * beta) + beta / 2;
        };
        auto const g_0 = [beta](auto const _x) {
            if (_x <= 1.0f - beta) return -1.0f;
            if (_x >= 1.0f + beta) return 1.0f;
            return (_x - 1.0f) / beta;
        };
        auto const func =
            f_0(x)
            + 2.0f * (1.0f - beta) / (l * pi) * std::sin(l * pi / 2.0f * x);
        auto const grad = g_0(x) + (1.0f - beta) * std::cos(l * pi / 2.0f * x);
        return std::make_pair(func, grad);
    };
    auto const [func_0, grad_0] = value_and_gradient(0.0f);

    {
        ::LBFGS_NAMESPACE::param_type params;
        params.f_tol = 1e-1f;
        params.g_tol = 1e-1f;
        auto const [status, alpha, _, grad, num_f_evals] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient, params, func_0,
                                           grad_0, /*alpha_0=*/0.001f);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        REQUIRE(alpha == Approx(1.0f).epsilon(1e-2f));
        REQUIRE(grad == Approx(-5.1e-5f).margin(2e-5f));
        REQUIRE(num_f_evals == 12);
    }

    {
        ::LBFGS_NAMESPACE::param_type params;
        params.f_tol = 1e-1f;
        params.g_tol = 1e-1f;
        auto const [status, alpha, _, grad, num_f_evals] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient, params, func_0,
                                           grad_0, /*alpha_0=*/0.1f);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        REQUIRE(alpha == Approx(1.0f).epsilon(1e-2f));
        REQUIRE(grad == Approx(-1.9e-4f).margin(2e-5f));
        REQUIRE(num_f_evals == 12);
    }

    {
        ::LBFGS_NAMESPACE::param_type params;
        params.f_tol = 1e-1f;
        params.g_tol = 1e-1f;
        auto const [status, alpha, _, grad, num_f_evals] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient, params, func_0,
                                           grad_0, /*alpha_0=*/10.0f);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        REQUIRE(alpha == Approx(1.0f).epsilon(1e-2f));
        REQUIRE(grad == Approx(-2.0e-6f).margin(1e-5f));
        REQUIRE(num_f_evals == 10);
    }

    {
        ::LBFGS_NAMESPACE::param_type params;
        params.f_tol = 1e-1f;
        params.g_tol = 1e-1f;
        auto const [status, alpha, _, grad, num_f_evals] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient, params, func_0,
                                           grad_0, /*alpha_0=*/1000.0f);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        REQUIRE(alpha == Approx(1.0f).epsilon(1e-2f));
        REQUIRE(grad == Approx(-1.6e-5f).margin(5e-5f));
        REQUIRE(num_f_evals == 13);
    }
}

TEST_CASE("Function (5.4) β₁=0.001, β₂=0.001", "[line_search]")
{
    auto const value_and_gradient = [](auto const x) {
        static_assert(std::is_same_v<std::remove_const_t<decltype(x)>, float>);
        auto const gamma = [](auto const beta) {
            return std::sqrt(1.0f + beta * beta) - beta;
        };
        constexpr auto beta_1  = 0.001f;
        constexpr auto beta_2  = 0.001f;
        constexpr auto gamma_1 = 0.9990005f;
        constexpr auto gamma_2 = 0.9990005f;
        auto const     a = std::sqrt((1.0f - x) * (1.0f - x) + beta_2 * beta_2);
        auto const     b = std::sqrt(x * x + beta_1 * beta_1);
        auto const     func = gamma_1 * a + gamma_2 * b;
        auto const     grad = gamma_2 * x / b + gamma_1 * (x - 1.0f) / a;
        return std::make_pair(func, grad);
    };
    auto const [func_0, grad_0] = value_and_gradient(0.0f);
    REQUIRE(grad_0 == Approx(-0.9990000005f).epsilon(1e-5f));
    REQUIRE(func_0 == Approx(1.0f).epsilon(1e-5f));

    {
        ::LBFGS_NAMESPACE::param_type params;
        params.f_tol = 1e-3f;
        params.g_tol = 1e-3f;
        auto const [status, alpha, _, grad, num_f_evals] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient, params, func_0,
                                           grad_0, /*alpha_0=*/0.001f);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        // REQUIRE(alpha == Approx(0.08f).epsilon(1e-2f));
        // REQUIRE(grad == Approx(-6.9e-5f).margin(1e-5f));
        // REQUIRE(num_f_evals == 4);
    }

    {
        ::LBFGS_NAMESPACE::param_type params;
        params.f_tol = 1e-3f;
        params.g_tol = 1e-3f;
        auto const [status, alpha, _, grad, num_f_evals] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient, params, func_0,
                                           grad_0, /*alpha_0=*/0.1f);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        // REQUIRE(alpha == Approx(0.08f).epsilon(1e-2f));
        // REQUIRE(grad == Approx(-6.9e-5f).margin(1e-5f));
        // REQUIRE(num_f_evals == 4);
    }

    {
        ::LBFGS_NAMESPACE::param_type params;
        params.f_tol = 1e-3f;
        params.g_tol = 1e-3f;
        auto const [status, alpha, _, grad, num_f_evals] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient, params, func_0,
                                           grad_0, /*alpha_0=*/10.0f);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        // REQUIRE(alpha == Approx(0.08f).epsilon(1e-2f));
        // REQUIRE(grad == Approx(-6.9e-5f).margin(1e-5f));
        // REQUIRE(num_f_evals == 4);
    }

    {
        ::LBFGS_NAMESPACE::param_type params;
        params.f_tol = 1e-3f;
        params.g_tol = 1e-3f;
        auto const [status, alpha, _, grad, num_f_evals] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient, params, func_0,
                                           grad_0, /*alpha_0=*/1000.0f);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        // REQUIRE(alpha == Approx(0.08f).epsilon(1e-2f));
        // REQUIRE(grad == Approx(-6.9e-5f).margin(1e-5f));
        // REQUIRE(num_f_evals == 4);
    }
}

TEST_CASE("Function (5.4) β₁=0.01, β₂=0.001", "[line_search]")
{
    auto const value_and_gradient = [](auto const x) {
        static_assert(std::is_same_v<std::remove_const_t<decltype(x)>, float>);
        auto const gamma = [](auto const beta) {
            return std::sqrt(1.0f + beta * beta) - beta;
        };
        constexpr auto beta_1  = 0.01f;
        constexpr auto beta_2  = 0.001f;
        constexpr auto gamma_1 = 0.9900499990f;
        constexpr auto gamma_2 = 0.9990005f;
        auto const     a = std::sqrt((1.0f - x) * (1.0f - x) + beta_2 * beta_2);
        auto const     b = std::sqrt(x * x + beta_1 * beta_1);
        auto const     func = gamma_1 * a + gamma_2 * b;
        auto const     grad = gamma_2 * x / b + gamma_1 * (x - 1.0f) / a;
        return std::make_pair(func, grad);
    };
    auto const [func_0, grad_0] = value_and_gradient(0.0f);
    REQUIRE(grad_0 == Approx(-0.9900495039f).epsilon(1e-5f));
    REQUIRE(func_0 == Approx(1.000040499f).epsilon(1e-5f));

    {
        ::LBFGS_NAMESPACE::param_type params;
        params.f_tol = 1e-3f;
        params.g_tol = 1e-3f;
        auto const [status, alpha, _, grad, num_f_evals] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient, params, func_0,
                                           grad_0, /*alpha_0=*/0.001f);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        REQUIRE(alpha == Approx(0.075f).epsilon(1e-2f));
        REQUIRE(grad == Approx(1.9e-4f).margin(5e-5f));
        // REQUIRE(num_f_evals == 6);
    }

    {
        ::LBFGS_NAMESPACE::param_type params;
        params.f_tol = 1e-3f;
        params.g_tol = 1e-3f;
        auto const [status, alpha, _, grad, num_f_evals] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient, params, func_0,
                                           grad_0, /*alpha_0=*/0.1f);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        REQUIRE(alpha == Approx(0.078f).epsilon(1e-2f));
        REQUIRE(grad == Approx(7.4e-4f).margin(2e-5f));
        REQUIRE(num_f_evals == 3);
    }

    {
        ::LBFGS_NAMESPACE::param_type params;
        params.f_tol = 1e-3f;
        params.g_tol = 1e-3f;
        auto const [status, alpha, _, grad, num_f_evals] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient, params, func_0,
                                           grad_0, /*alpha_0=*/10.0f);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        REQUIRE(alpha == Approx(0.073f).epsilon(1e-2f));
        REQUIRE(grad == Approx(-2.6e-4f).margin(2e-5f));
        REQUIRE(num_f_evals == 7);
    }

    {
        ::LBFGS_NAMESPACE::param_type params;
        params.f_tol = 1e-3f;
        params.g_tol = 1e-3f;
        auto const [status, alpha, _, grad, num_f_evals] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient, params, func_0,
                                           grad_0, /*alpha_0=*/1000.0f);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        REQUIRE(alpha == Approx(0.076f).epsilon(1e-2f));
        REQUIRE(grad == Approx(4.5e-4f).margin(2e-5f));
        REQUIRE(num_f_evals == 8);
    }
}

TEST_CASE("Function (5.4) β₁=0.001, β₂=0.01", "[line_search]")
{
    auto const value_and_gradient = [](auto const x) {
        static_assert(std::is_same_v<std::remove_const_t<decltype(x)>, float>);
        auto const gamma = [](auto const beta) {
            return std::sqrt(1.0f + beta * beta) - beta;
        };
        constexpr auto beta_1  = 0.001f;
        constexpr auto beta_2  = 0.01f;
        constexpr auto gamma_1 = 0.9990005f;
        constexpr auto gamma_2 = 0.9900499990f;
        auto const     a = std::sqrt((1.0f - x) * (1.0f - x) + beta_2 * beta_2);
        auto const     b = std::sqrt(x * x + beta_1 * beta_1);
        auto const     func = gamma_1 * a + gamma_2 * b;
        auto const     grad = gamma_2 * x / b + gamma_1 * (x - 1.0f) / a;
        return std::make_pair(func, grad);
    };
    auto const [func_0, grad_0] = value_and_gradient(0.0f);
    REQUIRE(grad_0 == Approx(-0.9989505539f).epsilon(1e-5f));
    REQUIRE(func_0 == Approx(1.000040499f).epsilon(1e-5f));

    {
        ::LBFGS_NAMESPACE::param_type params;
        params.f_tol = 1e-3f;
        params.g_tol = 1e-3f;
        auto const [status, alpha, _, grad, num_f_evals] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient, params, func_0,
                                           grad_0, /*alpha_0=*/0.001f);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        REQUIRE(alpha == Approx(0.93f).epsilon(1e-2f));
        REQUIRE(grad == Approx(5.2e-4f).margin(2e-3f));
        // REQUIRE(num_f_evals == 13);
    }

    {
        ::LBFGS_NAMESPACE::param_type params;
        params.f_tol = 1e-3f;
        params.g_tol = 1e-3f;
        auto const [status, alpha, _, grad, num_f_evals] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient, params, func_0,
                                           grad_0, /*alpha_0=*/0.1f);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        REQUIRE(alpha == Approx(0.93f).epsilon(1e-2f));
        REQUIRE(grad == Approx(8.4e-5f).margin(5e-4f));
        // REQUIRE(num_f_evals == 11);
    }

    {
        ::LBFGS_NAMESPACE::param_type params;
        params.f_tol = 1e-3f;
        params.g_tol = 1e-3f;
        auto const [status, alpha, _, grad, num_f_evals] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient, params, func_0,
                                           grad_0, /*alpha_0=*/10.0f);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        REQUIRE(alpha == Approx(0.92f).epsilon(1e-2f));
        REQUIRE(grad == Approx(-2.4e-4f).margin(2e-4f));
        // REQUIRE(num_f_evals == 8);
    }

    {
        ::LBFGS_NAMESPACE::param_type params;
        params.f_tol = 1e-3f;
        params.g_tol = 1e-3f;
        auto const [status, alpha, _, grad, num_f_evals] =
            ::LBFGS_NAMESPACE::line_search(value_and_gradient, params, func_0,
                                           grad_0, /*alpha_0=*/1000.0f);
        REQUIRE(status == ::LBFGS_NAMESPACE::status_t::success);
        REQUIRE(alpha == Approx(0.92f).epsilon(1e-2f));
        REQUIRE(grad == Approx(-3.2e-4f).margin(2e-4f));
        // REQUIRE(num_f_evals == 11);
    }
}

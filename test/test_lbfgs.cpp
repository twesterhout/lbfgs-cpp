#include "lbfgs.hpp"
#include <catch2/catch.hpp>

#include <iostream>

TEST_CASE("Test function 1", "[lbfgs]")
{
    // From Matlab's help page
    // https://www.mathworks.com/help/optim/ug/fminunc.html
    //
    // `f(x) = 3*x_1^2 + 2*x_1*x_2 + x_2^2 − 4*x_1 + 5*x_2`
    auto const value_and_gradient = [](auto const x, auto const g) {
        static_assert(std::is_same_v<std::remove_const_t<decltype(x)>,
                                     gsl::span<float const>>);
        static_assert(
            std::is_same_v<std::remove_const_t<decltype(g)>, gsl::span<float>>);
        auto const x1 = static_cast<double>(x[0]);
        auto const x2 = static_cast<double>(x[1]);
        auto const f_x =
            3.0 * x1 * x1 + 2.0 * x1 * x2 + x2 * x2 - 4.0 * x1 + 5.0 * x2;
        g[0] = static_cast<float>(6.0 * x1 + 2.0 * x2 - 4.0);
        g[1] = static_cast<float>(2.0 * x1 + 2.0 * x2 + 5.0);
        return f_x;
    };

    {
        std::array<float, 2>             x0 = {1.0, 1.0};
        ::LBFGS_NAMESPACE::lbfgs_param_t params;
        auto const                       r =
            ::LBFGS_NAMESPACE::minimize(value_and_gradient, params, x0);
        REQUIRE(r.status == ::LBFGS_NAMESPACE::status_t::success);
        REQUIRE(r.func == Approx(-16.3750));
        REQUIRE(r.num_iter <= 7);
        REQUIRE(x0[0] == Approx(2.25f));
        REQUIRE(x0[1] == Approx(-4.75f));
    }
}

TEST_CASE("Sphere function", "[lbfgs]")
{
    auto const value_and_gradient = [](auto const xs, auto const gs) {
        static_assert(std::is_same_v<std::remove_const_t<decltype(xs)>,
                                     gsl::span<float const>>);
        static_assert(std::is_same_v<std::remove_const_t<decltype(gs)>,
                                     gsl::span<float>>);
        auto f_x = 0.0;
        for (auto i = size_t{0}; i < xs.size(); ++i) {
            auto const x = static_cast<double>(xs[i]);
            f_x += x * x;
            gs[i] = 2.0f * xs[i];
        }
        return f_x;
    };

    {
        for (auto& x0 : std::vector<std::array<float, 5>>{
                 {1.0f, 1.0f, -8.0f, 1.3f, -0.002f},
                 {-1000.0f, 0.0f, 3.1293f, 9.0f, 9.0f},
             }) {
            ::LBFGS_NAMESPACE::lbfgs_param_t params;
            auto const                       r =
                ::LBFGS_NAMESPACE::minimize(value_and_gradient, params, x0);
            REQUIRE(r.status == ::LBFGS_NAMESPACE::status_t::success);
            REQUIRE(r.func == Approx(0));
            // One function evaluation for estimating gradients, then quadratic
            // interpolation should kick in and we get to the final answer in
            // one jump.
            REQUIRE(r.num_iter <= 2);
            for (auto const x : x0) {
                REQUIRE(x == Approx(0.0f));
            }
        }
    }
}

#if 0
TEST_CASE("Ackley function", "[lbfgs]")
{
    auto const value_and_gradient = [](auto const xs, auto const gs) {
        static_assert(std::is_same_v<std::remove_const_t<decltype(xs)>,
                                     gsl::span<float const>>);
        static_assert(std::is_same_v<std::remove_const_t<decltype(gs)>,
                                     gsl::span<float>>);
        REQUIRE(xs.size() == 2);
        REQUIRE(gs.size() == 2);
        auto const x = static_cast<double>(xs[0]);
        auto const y = static_cast<double>(xs[1]);
        auto       f_x =
            -20.0 * std::exp(-0.2 * std::sqrt(0.5 * (x * x + y * y)))
            - std::exp(0.5
                       * (std::cos(2.0 * M_PI * x) + std::cos(2.0 * M_PI * y)))
            + M_E + 20.0;
        gs[0] = static_cast<float>(
            M_PI * std::sin(2 * M_PI * x)
                * std::exp(0.5 * std::cos(2 * M_PI * x)
                           + 0.5 * std::cos(2 * M_PI * y))
            + 4 * x * std::exp(-std::sqrt(2 * x * x + 2 * y * y) / 10.0)
                  / std::sqrt(2 * x * x + 2 * y * y));
        gs[1] = static_cast<float>(
            M_PI * std::sin(2 * M_PI * y)
                * std::exp(0.5 * std::cos(2 * M_PI * y)
                           + 0.5 * std::cos(2 * M_PI * x))
            + 4 * y * std::exp(-std::sqrt(2 * y * y + 2 * x * x) / 10.0)
                  / std::sqrt(2 * y * y + 2 * x * x));
        LBFGS_TRACE("f(%e, %e) = %e, df/dx = [%e, %e]\n", x, y, f_x, gs[0],
                    gs[1]);
        return f_x;
    };

    {
        for (auto& x0 : std::vector<std::array<float, 2>>{
                 {0.1f, 0.3f},
             }) {
            ::LBFGS_NAMESPACE::lbfgs_param_t params;
            auto const                       r =
                ::LBFGS_NAMESPACE::minimize(value_and_gradient, params, x0);
            REQUIRE(r.status == ::LBFGS_NAMESPACE::status_t::success);
            REQUIRE(r.func == Approx(0));
            for (auto const x : x0) {
                REQUIRE(x == Approx(0.0f));
            }
        }
    }
}
#endif

TEST_CASE("Rosenbrock function", "[lbfgs]")
{
    auto const value_and_gradient = [](auto const xs, auto const gs) {
        static_assert(std::is_same_v<std::remove_const_t<decltype(xs)>,
                                     gsl::span<float const>>);
        static_assert(std::is_same_v<std::remove_const_t<decltype(gs)>,
                                     gsl::span<float>>);
        auto f_x = 0.0;
        LBFGS_ASSERT(xs.size() > 1, LBFGS_BUG_MESSAGE);
        for (auto i = size_t{0}; i < xs.size() - 1; ++i) {
            auto const x  = static_cast<double>(xs[i]);
            auto const t1 = static_cast<double>(xs[i + 1]) - x * x;
            auto const t2 = 1.0 - x;
            f_x += 100 * t1 * t1 + t2 * t2;
        }
        {
            auto const x = static_cast<double>(xs[0]);
            gs[0]        = static_cast<float>(
                -400.0 * (static_cast<double>(xs[1]) - x * x) * x
                - 2.0 * (1.0 - x));
        }
        for (auto i = size_t{1}; i < xs.size() - 1; ++i) {
            auto const x = static_cast<double>(xs[i]);
            gs[i]        = static_cast<float>(
                200.0
                    * (x
                       - static_cast<double>(xs[i - 1])
                             * static_cast<double>(xs[i - 1]))
                - 400.0 * (static_cast<double>(xs[i + 1]) - x * x) * x
                - 2.0 * (1.0 - x));
        }
        {
            auto const x      = static_cast<double>(xs[xs.size() - 1]);
            gs[xs.size() - 1] = static_cast<float>(
                200.0
                * (x
                   - static_cast<double>(xs[xs.size() - 2])
                         * static_cast<double>(xs[xs.size() - 2])));
        }
        LBFGS_TRACE("f(%e, %e) = %e, df/dx = [%e, %e]\n", xs[0], xs[1], f_x,
                    gs[0], gs[1]);
        return f_x;
    };

    {
        for (auto& x0 : std::vector<std::array<float, 2>>{
                 {15.0f, 8.0f},
             }) {
            ::LBFGS_NAMESPACE::lbfgs_param_t params;
            auto const                       r =
                ::LBFGS_NAMESPACE::minimize(value_and_gradient, params, x0);
            REQUIRE(r.status == ::LBFGS_NAMESPACE::status_t::success);
            REQUIRE(r.func == Approx(0).margin(1.0e-10));
            // REQUIRE(r.num_iter <= 2);
            for (auto const x : x0) {
                REQUIRE(x == Approx(1.0f));
            }
        }
    }
}

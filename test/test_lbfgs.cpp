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
        // LBFGS_TRACE("f(%e, %e) = %e, df/dx = [%e, %e]\n", xs[0], xs[1], f_x,
        //             gs[0], gs[1]);
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

TEST_CASE("Beale function", "[lbfgs]")
{
    // In Maple:
    // f(x, y) := (1.5 - x + x * y)^2 + (2.25 - x + x * y^2)^2 + (2.625 - x + x * y^3)^2;
    auto const value_and_gradient = [](auto const xs, auto const gs) {
        static_assert(std::is_same_v<std::remove_const_t<decltype(xs)>,
                                     gsl::span<float const>>);
        static_assert(std::is_same_v<std::remove_const_t<decltype(gs)>,
                                     gsl::span<float>>);
        LBFGS_ASSERT(xs.size() == 2, LBFGS_BUG_MESSAGE);
        auto const x   = static_cast<double>(xs[0]);
        auto const y   = static_cast<double>(xs[1]);
        auto const t1  = (1.5 - x + x * y);
        auto const t2  = (2.25 - x + x * y * y);
        auto const t3  = (2.625 - x + x * y * y * y);
        auto const f_x = t1 * t1 + t2 * t2 + t3 * t3;
        gs[0] =
            static_cast<float>(2.0 * t1 * (y - 1.0) + 2.0 * t2 * (y * y - 1.0)
                               + 2.0 * t3 * (y * y * y - 1.0));
        gs[1] = static_cast<float>(2.0 * t1 * x + 4.0 * t2 * x * y
                                   + 6.0 * t3 * x * y * y);
        return f_x;
    };
    {
        for (auto& x0 : std::vector<std::array<float, 2>>{
                 {2.5f, -1.0f}, {8.1f, 1.0f}, {5.0f, 5.0f}, {1.01f, 0.5001f}}) {
            ::LBFGS_NAMESPACE::lbfgs_param_t params;
            auto const                       r =
                ::LBFGS_NAMESPACE::minimize(value_and_gradient, params, x0);
            REQUIRE(r.status == ::LBFGS_NAMESPACE::status_t::success);
            REQUIRE(r.func == Approx(0).margin(1.0e-10));
            // REQUIRE(r.num_iter <= 2);
            REQUIRE(x0[0] == Approx(3.0f));
            REQUIRE(x0[1] == Approx(0.5f));
        }
    }
}

TEST_CASE("Goldstein-Price function", "[lbfgs]")
{
    auto const value_and_gradient = [](auto const xs, auto const gs) {
        static_assert(std::is_same_v<std::remove_const_t<decltype(xs)>,
                                     gsl::span<float const>>);
        static_assert(std::is_same_v<std::remove_const_t<decltype(gs)>,
                                     gsl::span<float>>);
        LBFGS_ASSERT(xs.size() == 2, LBFGS_BUG_MESSAGE);
        auto const x  = static_cast<double>(xs[0]);
        auto const y  = static_cast<double>(xs[1]);
        auto const t1 = (x + y + 1.0);
        auto const t2 = (19.0 - 14.0 * x + 3.0 * x * x - 14.0 * y + 6.0 * x * y
                         + 3.0 * y * y);
        auto const t3 = (2.0 * x - 3.0 * y);
        auto const t4 = 18.0 - 32.0 * x + 12.0 * x * x + 48.0 * y - 36.0 * x * y
                        + 27.0 * y * y;
        auto const f_x = (1.0 + t1 * t1 * t2) * (30.0 + t3 * t3 * t4);
        gs[0]          = static_cast<float>(
            (2.0 * t1 * t2 + t1 * t1 * (6.0 * x + 6.0 * y - 14.0))
                * (30.0 + t3 * t3 * t4)
            + (1.0 + t1 * t1 * t2)
                  * (4.0 * t3 * t4 + t3 * t3 * (24.0 * x - 36.0 * y - 32.0)));
        gs[1] = static_cast<float>(
            (2.0 * t1 * t2 + t1 * t1 * (6.0 * x + 6.0 * y - 14.0))
                * (30.0 + t3 * t3 * t4)
            + (1.0 + t1 * t1 * t2)
                  * (-6.0 * t3 * t4 + t3 * t3 * (-36.0 * x + 54.0 * y + 48.0)));
        LBFGS_TRACE("f(%e, %e) = %e, df/dx = [%e, %e]\n",
                    static_cast<double>(xs[0]), static_cast<double>(xs[1]), f_x,
                    static_cast<double>(gs[0]), static_cast<double>(gs[1]));
        return f_x;
    };

    {
        std::array<float, 2> xs = {1.8f, 0.2f};
        std::array<float, 2> gs;
        value_and_gradient(gsl::span<float const>{xs}, gsl::span<float>{gs});
    }
#if 1
    {
        for (auto& x0 :
             std::vector<std::array<float, 2>>{{0.5f, 0.5f}, {1.5f, -1.5f}}) {
            ::LBFGS_NAMESPACE::lbfgs_param_t params;
            auto const                       r =
                ::LBFGS_NAMESPACE::minimize(value_and_gradient, params, x0);
            REQUIRE(r.status == ::LBFGS_NAMESPACE::status_t::success);
            REQUIRE(r.func == Approx(3).margin(1.0e-10));
            // REQUIRE(r.num_iter <= 2);
            REQUIRE(x0[0] == Approx(0.0f).margin(1.0e-6));
            REQUIRE(x0[1] == Approx(-1.0f).margin(1.0e-6));
        }
        for (auto& x0 : std::vector<std::array<float, 2>>{{1.5f, 0.5f}}) {
            ::LBFGS_NAMESPACE::lbfgs_param_t params;
            auto const                       r =
                ::LBFGS_NAMESPACE::minimize(value_and_gradient, params, x0);
            REQUIRE(r.status == ::LBFGS_NAMESPACE::status_t::success);
            REQUIRE(r.func == Approx(84).margin(1.0e-10));
            // REQUIRE(r.num_iter <= 2);
            REQUIRE(x0[0] == Approx(1.8f).margin(1.0e-6));
            REQUIRE(x0[1] == Approx(0.2f).margin(1.0e-6));
        }
    }
#endif
}

#if 0
TEST_CASE("Booth function", "[lbfgs]")
{
    auto const value_and_gradient = [](auto const xs, auto const gs) {
        static_assert(std::is_same_v<std::remove_const_t<decltype(xs)>,
                                     gsl::span<float const>>);
        static_assert(std::is_same_v<std::remove_const_t<decltype(gs)>,
                                     gsl::span<float>>);
        LBFGS_ASSERT(xs.size() == 2, LBFGS_BUG_MESSAGE);
        auto const x  = static_cast<double>(xs[0]);
        auto const y  = static_cast<double>(xs[1]);
        auto const t1 = (x + y + 1.0);
        auto const t2 = (19.0 - 14.0 * x + 3.0 * x * x - 14.0 * y + 6.0 * x * y
                         + 3.0 * y * y);
        auto const t3 = (2.0 * x - 3.0 * y);
        auto const t4 = 18.0 - 32.0 * x + 12.0 * x * x + 48.0 * y - 36.0 * x * y
                        + 27.0 * y * y;
        auto const f_x = (1.0 + t1 * t1 * t2) * (30.0 + t3 * t3 * t4);
        gs[0]          = static_cast<float>(
            (2.0 * t1 * t2 + t1 * t1 * (6.0 * x + 6.0 * y - 14.0))
                * (30 + t3 * t3 * t4)
            + (1.0 + t1 * t1 * t2)
                  * (4.0 * t3 * t4 + t3 * t3 * (24.0 * x - 36.0 * y - 32.0)));
        gs[1] = static_cast<float>(
            (2.0 * t1 * t2 + t1 * t1 * (6.0 * x + 6.0 * y - 14.0))
                * (30 + t3 * t3 * t4)
            + (1 + t1 * t1 * t2)
                  * (-6.0 * t3 * t4 + t3 * t3 * (-36.0 * x + 54.0 * y + 48.0)));
        LBFGS_TRACE("f(%e, %e) = %e, df/dx = [%e, %e]\n",
                    static_cast<double>(xs[0]), static_cast<double>(xs[1]), f_x,
                    static_cast<double>(gs[0]), static_cast<double>(gs[1]));
        return f_x;
    };
    {
        for (auto& x0 :
             std::vector<std::array<float, 2>>{{0.5f, 0.5f}, {1.5f, -1.5f}}) {
            ::LBFGS_NAMESPACE::lbfgs_param_t params;
            auto const                       r =
                ::LBFGS_NAMESPACE::minimize(value_and_gradient, params, x0);
            REQUIRE(r.status == ::LBFGS_NAMESPACE::status_t::success);
            REQUIRE(r.func == Approx(3).margin(1.0e-10));
            // REQUIRE(r.num_iter <= 2);
            REQUIRE(x0[0] == Approx(0.0f).margin(1.0e-6));
            REQUIRE(x0[1] == Approx(-1.0f).margin(1.0e-6));
        }
        for (auto& x0 : std::vector<std::array<float, 2>>{{1.5f, 0.5f}}) {
            ::LBFGS_NAMESPACE::lbfgs_param_t params;
            auto const                       r =
                ::LBFGS_NAMESPACE::minimize(value_and_gradient, params, x0);
            REQUIRE(r.status == ::LBFGS_NAMESPACE::status_t::success);
            REQUIRE(r.func == Approx(84).margin(1.0e-10));
            // REQUIRE(r.num_iter <= 2);
            REQUIRE(x0[0] == Approx(1.8f).margin(1.0e-6));
            REQUIRE(x0[1] == Approx(0.2f).margin(1.0e-6));
        }
    }
}
#endif

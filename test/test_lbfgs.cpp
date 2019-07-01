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
        std::cout << make_error_code(r.status).message() << '\n';
        REQUIRE(r.status == ::LBFGS_NAMESPACE::status_t::success);
        REQUIRE(r.func == Approx(-16.3750));
        REQUIRE(x0[0] == Approx(2.25f));
        REQUIRE(x0[1] == Approx(-4.75f));
    }
}

#include "lbfgs.hpp"
#include <iostream>

int main()
{
    constexpr size_t N = 10;
    static_assert(N % 2 == 0);
    std::vector<float> x0(N);

    for (auto i = size_t{0}; i < N; i += 2) {
        x0[i]     = -1.2f;
        x0[i + 1] = 1.0f;
    }

    auto const value_and_gradient = [](auto const x, auto const grad) {
        auto f_x = 0.0;
        for (auto i = size_t{0}; i < x.size(); i += 2) {
            auto const a  = static_cast<double>(x[i]);
            auto const b  = static_cast<double>(x[i + 1]);
            auto const t1 = 1.0 - a;
            auto const t2 = 10.0 * (b - a * a);
            grad[i + 1]   = static_cast<float>(20.0 * t2);
            grad[i]       = static_cast<float>(-2.0 * (20.0 * a * t2 + t1));
            f_x += t1 * t1 + t2 * t2;
        }
        return f_x;
    };

    auto result = tcm::lbfgs::minimize(value_and_gradient,
                                       tcm::lbfgs::lbfgs_param_t{}, {x0});

    std::cerr << result.func << '\n';
    return 0;
}

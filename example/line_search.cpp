
#include "line_search.hpp"
#include "morethuente.h"

#include <iostream>

int main()
{
#if 0
    auto const func = [](auto const x) { return std::pow(x - 2.0f, 2.0f); };
    auto const grad = [](auto const x) { return 2.0f * (x - 2.0f); };

    auto const x_0    = -1.0f;
    auto const func_0 = func(x_0);
    auto const grad_0 = grad(x_0);
    auto const value_and_gradient =
        [func, grad, x_0](auto const step) -> std::pair<float, float> {
        return {func(x_0 + step), grad(x_0 + step)};
    };

    auto result = tcm::MoreThuente<void, 0>::line_search(value_and_gradient,
                                                         func_0, grad_0, 1.0f);
    std::cout << static_cast<int>(result.first) << '\t' << result.second
              << '\n';
#elif 0
    auto const value_and_gradient =
        [](auto const x) -> std::pair<float, float> {
        constexpr auto sigma = 0.15f;
        auto const     value = x <= 1.0f
                               ? (0.5f * (1.0f - sigma) * std::pow(x, 2.0f) - x)
                               : (0.5f * (sigma - 1.0f) - sigma * x);
        auto const grad = x <= 1.0f ? ((1.0f - sigma) * x - 1.0f) : (-sigma);
        return {value, grad};
    };
    auto const [func_0, grad_0] = value_and_gradient(0.0f);

    auto result_2 =
        cppoptlib::MoreThuente<void, 0>::linesearch(value_and_gradient, 1.0f);
    std::cout << result_2 << '\n';

    auto result = tcm::MoreThuente<void, 0>::line_search(value_and_gradient,
                                                         func_0, grad_0, 1.0f);
    std::cout << static_cast<int>(result.first) << '\t' << result.second
              << '\n';

#else
    auto const value_and_gradient = [](auto const x) {
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

    auto result_2 =
        cppoptlib::MoreThuente<void, 0>::linesearch(value_and_gradient, 0.1f);
    std::cout << result_2 << '\n';

    ::LBFGS_NAMESPACE::param_type params;
    params.f_tol = 1e-3f;
    params.g_tol = 1e-3f;
    auto const [status, alpha, func, grad, num_f_evals] =
        ::LBFGS_NAMESPACE::line_search(value_and_gradient, params, func_0,
                                       grad_0, 0.1f);
    std::cout << static_cast<int>(status) << '\t' << alpha << '\n';

#endif
}

#include "lbfgs.hpp"
#include <iostream>

int main()
{
    constexpr size_t N = 10;
    static_assert(N % 2 == 0);
    std::vector<float> x(N);

    for (auto i = size_t{0}; i < N; i += 2) {
        x[i]     = -1.2f;
        x[i + 1] = 1.0f;
    }

    auto const value_and_gradient = [](auto const x, auto const grad) {
        auto f_x = 0.0;
        for (auto i = size_t{0}; i < x.size(); i += 2) {
            auto const t1 = 1.0 - x[i];
            auto const t2 = 10.0 * (x[i + 1] - x[i] * x[i]);
            grad[i + 1]   = 20.0 * t2;
            grad[i]       = -2.0 * (x[i] * grad[i + 1] + t1);
            f_x += t1 * t1 + t2 * t2;
        }
        return f_x;
    };

    tcm::lbfgs::lbfgs_buffers_t buffers(10, 5, 0);
    auto                        state = buffers.make_state();

    std::copy(std::begin(x), std::end(x), std::begin(state.current.x));
    state.current.value =
        value_and_gradient(state.current.x, state.current.grad);
    std::cerr << "initial value = " << state.current.value << '\n';

    tcm::lbfgs::print_span("x    = ", state.current.x);
    tcm::lbfgs::print_span("grad = ", state.current.grad);
    tcm::lbfgs::print_span("x    = ", state.previous.x);
    tcm::lbfgs::print_span("grad = ", state.previous.grad);
    tcm::lbfgs::print_span("dir  = ", state.direction);

    tcm::lbfgs::minimize(value_and_gradient, tcm::lbfgs::lbfgs_param_t{},
                         state);

    std::cerr << state.current.value << '\n';
    return 0;
}

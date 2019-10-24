# LBFGS++
A readable implementation of L-BFGS method in modern C++


## Motivation

[L-BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS) is a pretty simple
algorithm. Nowadays, one can find an explanation of it in pretty much any
textbook on numerical optimisation (e.g. I used [this
book](https://www.springer.com/gp/book/9780387303031) by J.Nocedal and S.Wright
as a reference). Like most iterative optimisation algorithms it amounts to

* figuring out the direction in which to step, and
* calculating the step size

repeating these steps until convergence is reached. The tricky part is
calculation of the step size. [Line
search](https://en.wikipedia.org/wiki/Line_search) algorithms are usually used
for this. There are two line search algorithms which I am aware of which are
considered state-of-the-art:
[More-Thuente](https://dl.acm.org/citation.cfm?id=192132) and
[Hager-Zhang](https://dl.acm.org/citation.cfm?id=1132979).

I decided to look for an implementation of More-Thuente line search algorithm
first since it has been around since 1994. However, to my surprise, there
appears to only be [*one*
implementation](https://www.cs.umd.edu/users/oleary/software/). All other
implementations (e.g.
[rconjgrad](https://github.com/jlmelville/rconjgrad/blob/master/R/cvsrch.R),
[Trilinos](https://github.com/trilinos/Trilinos/blob/master/packages/nox/src/NOX_LineSearch_MoreThuente.C),
[LineSearches.jl](https://github.com/JuliaNLSolvers/LineSearches.jl/blob/master/src/morethuente.jl),
[liblbfgs](https://github.com/chokkan/liblbfgs/blob/master/lib/lbfgs.c),
[CppNumericalSolvers](https://github.com/PatWie/CppNumericalSolvers/blob/master/include/cppoptlib/linesearch/morethuente.h))
use the exact same code except for the translation from MatLab to their
language of choice.

It is also quite difficult to see how the code corresponds to the original
paper. So I decided to write an implementation which it simple enough so that
it could be understood by anyone who knows a bit of `C++17` and has read the
paper.

## Installing

The preferred way to use this library is [CMake](https://cmake.org/) + [git
submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules).

// CppNumericalSolver
#ifndef MORETHUENTE_H_
#define MORETHUENTE_H_

#include <cmath>
#include <iostream>

namespace cppoptlib {

template <typename ProblemType, int Ord> class MoreThuente {

  public:
    using Scalar = float; // typename ProblemType::Scalar;
    // using TVector = typename ProblemType::TVector;

    /**
   * @brief use MoreThuente Rule for (strong) Wolfe conditiions
   * @details [long description]
   *
   * @param searchDir search direction for next update step
   * @param objFunc handle to problem
   *
   * @return step-width
   */

    template <class Function>
    static Scalar linesearch(Function&&   value_and_gradient,
                             const Scalar alpha_init = 1.0)
    {
        // assume step width
        Scalar ak = alpha_init;

        auto [fval, g] = value_and_gradient(0.0);

        cvsrch(value_and_gradient, fval, g, ak);
        return ak;
    }

#if 0
    static Scalar linesearch(const TVector& x, const TVector& searchDir,
                             ProblemType& objFunc,
                             const Scalar alpha_init = 1.0)
    {
        // assume step width
        Scalar ak = alpha_init;

        Scalar  fval = objFunc.value(x);
        TVector g    = x.eval();
        objFunc.gradient(x, g);

        TVector s  = searchDir.eval();
        TVector xx = x.eval();

        cvsrch(objFunc, xx, fval, g, ak, s);

        return ak;
    }
#endif

    template <class Function>
    static int cvsrch(Function&& value_and_gradient, Scalar f, Scalar dginit,
                      Scalar& stp)
    {
        // we rewrite this from MIN-LAPACK and some MATLAB code
        int          info   = 0;
        int          infoc  = 1;
        const Scalar xtol   = 1e-7f;
        const Scalar ftol   = 1e-3f;
        const Scalar gtol   = 1e-3f;
        const Scalar stpmin = 1e-8f;
        const Scalar stpmax = 1e8f;
        const Scalar xtrapf = 4;
        const int    maxfev = 20;
        int          nfev   = 0;

        if (dginit >= 0.0f) {
            // no descent direction
            // TODO: handle this case
            return -1;
        }

        bool brackt = false;
        bool stage1 = true;

        Scalar finit  = f;
        Scalar dgtest = ftol * dginit;
        Scalar width  = stpmax - stpmin;
        Scalar width1 = 2.0f * width;

        Scalar stx = 0.0f;
        Scalar fx  = finit;
        Scalar dgx = dginit;
        Scalar sty = 0.0f;
        Scalar fy  = finit;
        Scalar dgy = dginit;

        Scalar stmin;
        Scalar stmax;

        while (true) {

            // make sure we stay in the interval when setting min/max-step-width
            if (brackt) {
                stmin = std::min<Scalar>(stx, sty);
                stmax = std::max<Scalar>(stx, sty);
            }
            else {
                stmin = stx;
                stmax = stp + xtrapf * (stp - stx);
                assert(stmin <= stmax);
            }

            // Force the step to be within the bounds stpmax and stpmin.
            stp = std::max<Scalar>(stp, stpmin);
            stp = std::min<Scalar>(stp, stpmax);

            std::cerr << "stp = " << stp << " in [" << stmin << ", " << stmax
                      << "]\n";

            // Oops, let us return the last reliable values
            if ((brackt && ((stp <= stmin) || (stp >= stmax)))
                || (nfev >= maxfev - 1) || (infoc == 0)
                || (brackt && ((stmax - stmin) <= (xtol * stmax)))) {
                stp = stx;
            }

            // test new point
            auto [_f, dg] = value_and_gradient(stp);
            f             = _f;
            nfev++;
            Scalar ftest1 = finit + stp * dgtest;

            // all possible convergence tests
            if ((brackt & ((stp <= stmin) | (stp >= stmax))) | (infoc == 0))
                info = 6;

            if ((stp == stpmax) & (f <= ftest1) & (dg <= dgtest)) info = 5;

            if ((stp == stpmin) & ((f > ftest1) | (dg >= dgtest))) info = 4;

            if (nfev >= maxfev) info = 3;

            if (brackt & (stmax - stmin <= xtol * stmax)) info = 2;

            if ((f <= ftest1) & (fabs(dg) <= gtol * (-dginit))) info = 1;

            // terminate when convergence reached
            if (info != 0) return -1;

            if (stage1 & (f <= ftest1)
                & (dg >= std::min<Scalar>(ftol, gtol) * dginit)) {
                std::cerr << "stage1 = false\n";
                stage1 = false;
            }

            if (stage1 & (f <= fx) & (f > ftest1)) {
                std::cerr << "Using modified updating scheme\n";
                Scalar fm   = f - stp * dgtest;
                Scalar fxm  = fx - stx * dgtest;
                Scalar fym  = fy - sty * dgtest;
                Scalar dgm  = dg - dgtest;
                Scalar dgxm = dgx - dgtest;
                Scalar dgym = dgy - dgtest;

                cstep(stx, fxm, dgxm, sty, fym, dgym, stp, fm, dgm, brackt,
                      stmin, stmax, infoc);

                fx  = fxm + stx * dgtest;
                fy  = fym + sty * dgtest;
                dgx = dgxm + dgtest;
                dgy = dgym + dgtest;
            }
            else {
                std::cerr << "Using normal updating scheme\n";
                // this is ugly and some variables should be moved to the class scope
                cstep(stx, fx, dgx, sty, fy, dgy, stp, f, dg, brackt, stmin,
                      stmax, infoc);
            }
            std::cerr << "cstep returned " << infoc << '\n';

            if (brackt) {
                std::cerr << "Checking for width\n";
                if (std::abs(sty - stx) >= 0.66f * width1)
                    stp = stx + 0.5f * (sty - stx);
                width1 = width;
                width  = std::abs(sty - stx);
            }
        }

        return 0;
    }

    static int cstep(Scalar& stx, Scalar& fx, Scalar& dx, Scalar& sty,
                     Scalar& fy, Scalar& dy, Scalar& stp, Scalar& fp,
                     Scalar& dp, bool& brackt, Scalar& stpmin, Scalar& stpmax,
                     int& info)
    {
        info       = 0;
        bool bound = false;

        // Check the input parameters for errors.
        if ((brackt
             & ((stp <= std::min<Scalar>(stx, sty))
                | (stp >= std::max<Scalar>(stx, sty))))
            | (dx * (stp - stx) >= 0.0f) | (stpmax < stpmin)) {
            return -1;
        }

        Scalar sgnd = dp * (dx / std::abs(dx));

        Scalar stpf = 0;
        Scalar stpc = 0;
        Scalar stpq = 0;

        if (fp > fx) {
            info         = 1;
            bound        = true;
            Scalar theta = 3.0f * (fx - fp) / (stp - stx) + dx + dp;
            Scalar s     = std::max<Scalar>(theta, std::max<Scalar>(dx, dp));
            Scalar gamma =
                std::abs(s)
                * std::sqrt((theta / s) * (theta / s) - (dx / s) * (dp / s));
            if (stp < stx) gamma = -gamma;
            Scalar p = (gamma - dx) + theta;
            Scalar q = ((gamma - dx) + gamma) + dp;
            Scalar r = p / q;
            std::cerr << "θ = " << theta << ", s = " << s << ", γ = " << gamma
                      << ", p = " << p << ", q = " << q << '\n';
            stpc = stx + r * (stp - stx);
            stpq =
                stx
                + ((dx / ((fx - fp) / (stp - stx) + dx)) / 2.0f) * (stp - stx);
            std::cerr << "case_1: α_c = " << stpc << ", α_q = " << stpq << '\n';
            if (std::abs(stpc - stx) < std::abs(stpq - stx))
                stpf = stpc;
            else
                stpf = stpc + (stpq - stpc) / 2.0f;
            brackt = true;
        }
        else if (sgnd < 0.0) {
            info         = 2;
            bound        = false;
            Scalar theta = 3 * (fx - fp) / (stp - stx) + dx + dp;
            Scalar s     = std::max<Scalar>(theta, std::max<Scalar>(dx, dp));
            Scalar gamma =
                s * std::sqrt((theta / s) * (theta / s) - (dx / s) * (dp / s));
            if (stp > stx) gamma = -gamma;

            Scalar p = (gamma - dp) + theta;
            Scalar q = ((gamma - dp) + gamma) + dx;
            Scalar r = p / q;
            stpc     = stp + r * (stx - stp);
            stpq     = stp + (dp / (dp - dx)) * (stx - stp);
            if (std::abs(stpc - stp) > std::abs(stpq - stp))
                stpf = stpc;
            else
                stpf = stpq;
            brackt = true;
        }
        else if (std::abs(dp) < std::abs(dx)) {
            info         = 3;
            bound        = 1;
            Scalar theta = 3 * (fx - fp) / (stp - stx) + dx + dp;
            Scalar s     = std::max<Scalar>(theta, std::max<Scalar>(dx, dp));
            Scalar gamma =
                s
                * std::sqrt(std::max<Scalar>(static_cast<Scalar>(0.),
                                             (theta / s) * (theta / s)
                                                 - (dx / s) * (dp / s)));
            if (stp > stx) gamma = -gamma;
            Scalar p = (gamma - dp) + theta;
            Scalar q = (gamma + (dx - dp)) + gamma;
            Scalar r = p / q;
            std::cerr << "γ = " << gamma << ", r = " << r << '\n';
            if ((r < 0.0) & (gamma != 0.0f)) { stpc = stp + r * (stx - stp); }
            else if (stp > stx) {
                std::cerr << "setting stpc := " << stpmax << '\n';
                stpc = stpmax;
            }
            else {
                stpc = stpmin;
            }
            stpq = stp + (dp / (dp - dx)) * (stx - stp);
            std::cerr << "stpc = " << stpc << ", stpq = " << stpq << '\n';
            if (brackt) {
                if (std::abs(stp - stpc) < std::abs(stp - stpq)) {
                    stpf = stpc;
                }
                else {
                    stpf = stpq;
                }
            }
            else {
                if (std::abs(stp - stpc) > std::abs(stp - stpq)) {
                    stpf = stpc;
                }
                else {
                    stpf = stpq;
                }
            }
            std::cerr << "  => stpf = " << stpf << '\n';
        }
        else {
            info  = 4;
            bound = false;
            if (brackt) {
                Scalar theta = 3.0f * (fp - fy) / (sty - stp) + dy + dp;
                Scalar s = std::max<Scalar>(theta, std::max<Scalar>(dy, dp));
                Scalar gamma = s
                               * std::sqrt((theta / s) * (theta / s)
                                           - (dy / s) * (dp / s));
                if (stp > sty) gamma = -gamma;

                Scalar p = (gamma - dp) + theta;
                Scalar q = ((gamma - dp) + gamma) + dy;
                Scalar r = p / q;
                stpc     = stp + r * (sty - stp);
                stpf     = stpc;
            }
            else if (stp > stx)
                stpf = stpmax;
            else {
                stpf = stpmin;
            }
        }

        if (fp > fx) {
            sty = stp;
            fy  = fp;
            dy  = dp;
        }
        else {
            if (sgnd < 0.0f) {
                sty = stx;
                fy  = fx;
                dy  = dx;
            }

            stx = stp;
            fx  = fp;
            dx  = dp;
        }
        std::cerr << "Took case " << info << '\n';

        stpf = std::min<Scalar>(stpmax, stpf);
        stpf = std::max<Scalar>(stpmin, stpf);
        stp  = stpf;

        if (brackt && bound) {
            if (sty > stx) {
                stp = std::min<Scalar>(
                    stx + static_cast<Scalar>(0.66) * (sty - stx), stp);
            }
            else {
                stp = std::max<Scalar>(
                    stx + static_cast<Scalar>(0.66) * (sty - stx), stp);
            }
        }

        return 0;
    }
};

} // namespace cppoptlib

#endif /* MORETHUENTE_H_ */

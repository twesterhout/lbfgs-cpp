#pragma once

#include "config.hpp"

#include <gsl/gsl-lite.hpp>

auto blas_dot(gsl::span<float const> a, gsl::span<float const> b) noexcept
    -> double;
auto blas_nrm2(gsl::span<float const> a) noexcept -> double;
auto blas_scal(float c, gsl::span<float> a) noexcept -> void;
auto blas_axpy(float a, gsl::span<float const> x, gsl::span<float> y) noexcept
    -> void;

auto custom_dot(gsl::span<float const> a, gsl::span<float const> b) noexcept
    -> double;
auto custom_nrm2(gsl::span<float const> a) noexcept -> double;
auto custom_scal(float c, gsl::span<float> a) noexcept -> void;
auto custom_axpy(float a, gsl::span<float const> x, gsl::span<float> y) noexcept
    -> void;

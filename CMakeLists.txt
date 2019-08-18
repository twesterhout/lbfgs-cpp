# Copyright Tom Westerhout (c) 2019
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials provided
#       with the distribution.
#
#     * Neither the name of Tom Westerhout nor the names of other
#       contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.9)
cmake_policy(SET CMP0074 NEW)

project(lbfgs-cpp)

# detect if we're being bundled, disable test and install targets in that case
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(NOT_SUBPROJECT ON)
else()
    set(NOT_SUBPROJECT OFF)
endif()

include(CTest)
include(CheckCXXCompilerFlag)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to RelWithDebInfo, use \
-DCMAKE_BUILD_TYPE= to override."
    )
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()
string(TOUPPER "${CMAKE_BUILD_TYPE}" _LBFGS_BUILD_TYPE)

if("${_LBFGS_BUILD_TYPE}" STREQUAL "DEBUG")
    set(_LBFGS_ASSERTS_ARE_OKAY ON)
else()
    set(_LBFGS_ASSERTS_ARE_OKAY OFF)
endif()

if(BUILD_TESTING AND NOT_SUBPROJECT)
    set(_LBFGS_TESTS_ARE_OKAY ON)
    set(_LBFGS_EXAMPLES_ARE_OKAY ON)
else()
    set(_LBFGS_TESTS_ARE_OKAY OFF)
    set(_LBFGS_EXAMPLES_ARE_OKAY ON)
endif()

option(LBFGS_USE_VALGRIND "Perform tests with Valgrind" OFF)
option(LBFGS_USE_BLAS "Use BLAS for 1D vector operations" OFF)
option(LBFGS_BUILD_TESTING "Build tests" ${_LBFGS_TESTS_ARE_OKAY})
option(LBFGS_BUILD_EXAMPLES "Build examples" ${_LBFGS_EXAMPLES_ARE_OKAY})
option(LBFGS_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(LBFGS_ENABLE_COVERAGE "Generate coverage for codecov.io" OFF)
option(LBFGS_INSTALL_DOCS "Install documentation alongside library" ON)
option(LBFGS_DEBUG "Include assertions" ${_LBFGS_ASSERTS_ARE_OKAY})


# Warnings
################################################################################
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(LBFGS_WARNING_FLAGS -Weverything -Wno-c++98-compat -Wno-c++98-compat-pedantic
        -Wno-c++98-c++11-c++14-compat -Wno-unused -Wno-documentation-unknown-command
        -Wno-float-equal -Wno-padded
        )
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(LBFGS_WARNING_FLAGS -pedantic -W -Wall -Wextra -Wcast-align -Wcast-qual
        -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self
        -Wlogical-op -Wmissing-declarations -Wmissing-include-dirs -Wnoexcept
        -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow
        -Wsign-conversion -Wsign-promo -Wstrict-null-sentinel
        -Wstrict-overflow=1 -Wswitch-default -Wundef -Wno-unused -Wno-noexcept)
endif()

find_package(Git)
if(GIT_FOUND)
    # Update submodules
    execute_process(
        COMMAND ${GIT_EXECUTABLE} submodule update --init -- third_party/gsl-lite
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    if (LBFGS_BUILD_TESTING)
        execute_process(
            COMMAND ${GIT_EXECUTABLE} submodule update --init -- third_party/Catch2
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    endif()
else()
    message(WARNING "Git not found so submodules were not updated. Expect problems")
endif()

set(lbfgs-cpp_HEADERS
    "include/lbfgs/config.hpp"
    "include/lbfgs/line_search.hpp"
    "include/lbfgs/lbfgs.hpp")

set(lbfgs-cpp_SOURCES
    "src/lbfgs.cpp")

function(setup_lbfgs_target MY_TARGET)
    target_compile_features(${MY_TARGET} PUBLIC cxx_std_17)
    target_compile_options(${MY_TARGET} PRIVATE "${LBFGS_WARNING_FLAGS}")
    if (LBFGS_DEBUG)
        target_compile_definitions(${MY_TARGET} PUBLIC LBFGS_DEBUG=1)
    endif()

    check_cxx_compiler_flag("-march=native" LBFGS_COMPILER_OPT_NATIVE_SUPPORTED)
    if(LBFGS_COMPILER_OPT_NATIVE_SUPPORTED)
        message(STATUS "[lbfgs-cpp] Using -march=native for ${MY_TARGET}.")
        target_compile_options(${MY_TARGET} PRIVATE "-march=native")
    endif()

    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        target_compile_options(${MY_TARGET} PUBLIC -fcolor-diagnostics)
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        target_compile_options(${MY_TARGET} PUBLIC -fdiagnostics-color=always)
    endif()
endfunction()

# Main target
add_library(lbfgs ${lbfgs-cpp_HEADERS}
                  ${lbfgs-cpp_SOURCES})
target_include_directories(lbfgs PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
setup_lbfgs_target(lbfgs)
check_cxx_compiler_flag("-fvisibility=hidden" LBFGS_COMPILER_OPT_HIDDEN_SUPPORTED)
if(LBFGS_COMPILER_OPT_HIDDEN_SUPPORTED)
    message(STATUS "[lbfgs-cpp] Using -fvisibility=hidden.")
    target_compile_options(lbfgs PRIVATE "-fvisibility=hidden")
endif()
if (LBFGS_USE_BLAS)
    find_package(CBLAS REQUIRED)
    target_compile_definitions(lbfgs INTERFACE LBFGS_USE_BLAS=1)
    target_link_libraries(lbfgs PUBLIC ${CBLAS_LIBRARIES})
    target_include_directories(lbfgs PUBLIC ${CBLAS_INCLUDE_DIR})
endif()

if (NOT TARGET gsl::gsl-lite)
    add_subdirectory(third_party/gsl-lite)
    if ("${_LBFGS_BUILD_TYPE}" STREQUAL "DEBUG")
        target_compile_definitions(lbfgs PUBLIC
            gsl_CONFIG_CONTRACT_VIOLATION_CALLS_HANDLER=1)
    else()
        target_compile_definitions(lbfgs PUBLIC
            gsl_CONFIG_CONTRACT_LEVEL_OFF=1)
        message(STATUS "[lbfgs-cpp] Disabling gsl-lite contracts")
    endif()
endif()
target_link_libraries(lbfgs PUBLIC gsl::gsl-lite)
target_compile_definitions(lbfgs PUBLIC LBFGS_USE_GSL_LITE=1)

if (LBFGS_BUILD_TESTING)
    if (NOT TARGET Catch2::Catch2)
        option(CATCH_BUILD_TESTING OFF)
        option(CATCH_BUILD_EXAMPLES OFF)
        option(CATCH_BUILD_EXTRA_TESTS OFF)
        add_subdirectory(third_party/Catch2)
    endif()
    enable_testing()
endif()

# if (TRUE)
#     target_compile_options(line_search PUBLIC "-fprofile-instr-generate" "-fcoverage-mapping")
#     target_link_libraries(line_search PUBLIC "-fprofile-instr-generate" "-fcoverage-mapping")
# endif()

if (LBFGS_BUILD_EXAMPLES)
    add_executable(line_search example/line_search.cpp)
    target_link_libraries(line_search PUBLIC lbfgs)

    add_executable(lbfgs_example example/lbfgs.cpp)
    target_link_libraries(lbfgs_example PUBLIC lbfgs)

    # add_subdirectory(examples)
endif()

if (LBFGS_BUILD_TESTING)
    add_subdirectory(test)
endif()
# 
# if (LBFGS_BUILD_BENCHMARKS)
#     add_subdirectory(bench)
# endif()

add_library(lbfgs-cpp::lbfgs ALIAS lbfgs)

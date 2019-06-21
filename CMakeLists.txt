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

# detect if we're being bundled, disable test and install targets in that case
if(NOT DEFINED PROJECT_NAME)
  set(NOT_SUBPROJECT ON)
endif()

project(lbfgs-cpp)

include(CTest)
include(CheckCXXCompilerFlag)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_EXTENSIONS OFF)

option(LBFGS_USE_VALGRIND "Perform tests with Valgrind" OFF)
option(LBFGS_BUILD_TESTING "Build tests" ON)
option(LBFGS_BUILD_EXAMPLES "Build examples" OFF)
option(LBFGS_ENABLE_COVERAGE "Generate coverage for codecov.io" ON)
option(LBFGS_INSTALL_DOCS "Install documentation alongside library" ON)
option(LBFGS_DEBUG "Include assertions" ON)

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to RelWithDebInfo, use \
-DCMAKE_BUILD_TYPE= to override."
    )
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_OPT_NATIVE_SUPPORTED)

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
        -Wstrict-overflow=1 -Wswitch-default -Wundef -Wno-unused)
endif()

execute_process(COMMAND git submodule update --init -- third_party/gsl-lite
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
add_library(gsl-lite INTERFACE)
target_include_directories(gsl-lite INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/gsl-lite/include)
string(TOUPPER "${CMAKE_BUILD_TYPE}" _TCM_BUILD_TYPE)
if("${CMAKE_BUILD_TYPE}" STREQUAL "DEBUG")
    target_compile_definitions(gsl-lite INTERFACE
        gsl_CONFIG_CONTRACT_VIOLATION_CALLS_HANDLER=1)
else()
    target_compile_definitions(gsl-lite INTERFACE
        gsl_CONFIG_CONTRACT_LEVEL_OFF=1)
    message(STATUS "Disabling GSL contracts")
endif()


execute_process(COMMAND git submodule update --init -- third_party/SG14
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
add_library(SG14 INTERFACE)
target_include_directories(SG14 INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/SG14)

find_package(BLAS REQUIRED)

add_library(Common INTERFACE)
target_include_directories(Common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(Common INTERFACE cxx_std_17)
target_compile_options(Common INTERFACE "${LBFGS_WARNING_FLAGS}")
target_link_libraries(Common INTERFACE gsl-lite)
if (LBFGS_DEBUG)
    target_compile_definitions(Common INTERFACE LBFGS_DEBUG=1)
endif()

add_library(lbfgs SHARED src/lbfgs.cpp)
target_link_libraries(lbfgs Common)


CHECK_CXX_COMPILER_FLAG("-fvisibility=hidden" LBFGS_COMPILER_OPT_HIDDEN_SUPPORTED)
if(LBFGS_COMPILER_OPT_HIDDEN_SUPPORTED)
    message(STATUS "[lbfgs-cpp] Using -fvisibility=hidden.")
    target_compile_options(Common INTERFACE "-fvisibility=hidden")
endif()

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    target_compile_options(Common INTERFACE -fcolor-diagnostics)
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    target_compile_options(Common INTERFACE -fdiagnostics-color=always)
endif()


add_executable(line_search example/line_search.cpp)
target_link_libraries(line_search PUBLIC lbfgs)

add_executable(lbfgs_example example/lbfgs.cpp)
target_link_libraries(lbfgs_example PUBLIC lbfgs ${BLAS_LIBRARIES})

# if (TRUE)
#     target_compile_options(line_search PUBLIC "-fprofile-instr-generate" "-fcoverage-mapping")
#     target_link_libraries(line_search PUBLIC "-fprofile-instr-generate" "-fcoverage-mapping")
# endif()





if (BUILD_TESTING AND LBFGS_BUILD_TESTING AND NOT_SUBPROJECT)
    execute_process(COMMAND git submodule update --init -- third_party/Catch2
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    option(CATCH_BUILD_TESTING OFF)
    option(CATCH_BUILD_EXAMPLES OFF)
    option(CATCH_BUILD_EXTRA_TESTS OFF)
    add_subdirectory(third_party/Catch2)

    enable_testing()
    add_subdirectory(test)
endif()


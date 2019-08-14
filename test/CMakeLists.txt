
add_executable(run_tests catch2_main.cpp
                         test_line_search.cpp
                         test_lbfgs.cpp
)
target_link_libraries(run_tests PUBLIC lbfgs Catch2::Catch2)
# target_code_coverage(run_tests AUTO ALL)

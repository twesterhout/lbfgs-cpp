
add_executable(run_tests catch2_main.cpp
                         test_line_search.cpp
                         test_lbfgs.cpp
)
setup_lbfgs_target(run_tests)
target_link_libraries(run_tests PUBLIC lbfgs Catch2::Catch2)
add_test(NAME all_tests COMMAND $<TARGET_FILE:run_tests>)
# target_code_coverage(run_tests AUTO ALL)

# 注释掉原来的下载和构建过程
# configure_file(CMakeLists.txt.in pybind11-download/CMakeLists.txt)
# execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
#     RESULT_VARIABLE result
#     WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/pybind11-download )
# if(result)
#     message(FATAL_ERROR "CMake step for pybind11 failed: ${result}")
# endif()
# execute_process(COMMAND ${CMAKE_COMMAND} --build .
#     RESULT_VARIABLE result
#     WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/pybind11-download )
# if(result)
#     message(FATAL_ERROR "Build step for pybind11 failed: ${result}")
# endif()

# 指定 pybind11 的本地路径
set(PYBIND11_DIR "${CMAKE_CURRENT_SOURCE_DIR}/pybind11-src")
add_subdirectory(${PYBIND11_DIR} ${CMAKE_BINARY_DIR}/pybind11-build)

# 设置 pybind11 的 include 路径
set(PYBIND11_INCLUDE
    ${PYBIND11_DIR}/include/
    CACHE INTERNAL "")

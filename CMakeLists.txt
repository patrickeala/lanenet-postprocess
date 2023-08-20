cmake_minimum_required(VERSION 3.10)

project(pat)

set(CMAKE_CXX_STANDARD 11)

# Find the OpenCV package
find_package(OpenCV REQUIRED)

# Set the paths to the cnpy library
set(CNPY_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/source)
set(CNPY_LIBRARY ${CMAKE_SOURCE_DIR}/source/cnpy.cpp) # Adjust the library file accordingly

# Add the executable
add_executable(main
    main.cpp
    source/PostProcess.cpp
    source/kdtree.cpp
    source/cnpy.cpp
)

# Include directories for header files
target_include_directories(main PRIVATE source ${CNPY_INCLUDE_DIR})

# Link the executable with the OpenCV, cnpy, and zlib libraries
target_link_libraries(main PRIVATE ${OpenCV_LIBS} -lcnpy -lz)
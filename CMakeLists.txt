cmake_minimum_required(VERSION 3.0.2)
project(pf_project)

add_compile_options(-std=c++17)

find_package(catkin REQUIRED COMPONENTS roscpp tf)
catkin_package()

include_directories(include ${catkin_INCLUDE_DIRS})

add_executable(likelihood_field_particle_filter src/likelihood_field_particle_filter.cpp)
target_link_libraries(likelihood_field_particle_filter ${catkin_LIBRARIES})
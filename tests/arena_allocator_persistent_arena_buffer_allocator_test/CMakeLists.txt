cmake_minimum_required(VERSION 3.12)

project(arena_allocator_persistent_arena_buffer_allocator_test C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

add_executable(arena_allocator_persistent_arena_buffer_allocator_test "")

target_include_directories(arena_allocator_persistent_arena_buffer_allocator_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/arena_allocator_persistent_arena_buffer_allocator_test
)

set_target_properties(
  arena_allocator_persistent_arena_buffer_allocator_test
  PROPERTIES
  COMPILE_FLAGS -Os
  COMPILE_FLAGS -fno-rtti
  COMPILE_FLAGS -fno-exceptions
  COMPILE_FLAGS -fno-threadsafe-statics
  COMPILE_FLAGS -nostdlib
)

pico_enable_stdio_usb(arena_allocator_persistent_arena_buffer_allocator_test 1)
pico_enable_stdio_uart(arena_allocator_persistent_arena_buffer_allocator_test 0)

target_sources(arena_allocator_persistent_arena_buffer_allocator_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/arena_allocator_persistent_arena_buffer_allocator_test/persistent_arena_buffer_allocator_test.cpp

)

target_link_libraries(
  arena_allocator_persistent_arena_buffer_allocator_test
  pico-tflmicro
  pico-tflmicro_test
)

pico_add_extra_outputs(arena_allocator_persistent_arena_buffer_allocator_test)

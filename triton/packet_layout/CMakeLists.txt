add_library(LAQSTritonPacketLayout SHARED PacketLayoutPlugin.cpp)
add_dependencies(LAQSTritonPacketLayout TritonTableGen TritonGPUTableGen)
target_link_libraries(LAQSTritonPacketLayout PRIVATE triton)
target_compile_options(LAQSTritonPacketLayout PRIVATE
  -fvisibility=hidden ${TRITON_DISABLE_EH_RTTI_FLAGS})
if(DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
  set_target_properties(LAQSTritonPacketLayout PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/../plugins")
endif()

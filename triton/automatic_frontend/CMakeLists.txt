add_library(LAQSTritonAccessManifest SHARED AccessManifestPlugin.cpp)
add_dependencies(LAQSTritonAccessManifest
  TritonTableGen
  TritonGPUTableGen
)
target_link_libraries(LAQSTritonAccessManifest PRIVATE triton)
target_compile_options(LAQSTritonAccessManifest PRIVATE
  -fvisibility=hidden
  ${TRITON_DISABLE_EH_RTTI_FLAGS}
)

if(DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
  set_target_properties(LAQSTritonAccessManifest PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/../plugins"
  )
endif()

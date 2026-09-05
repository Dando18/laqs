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

# Keep the runtime layout realization pass in a separate library.  Triton pass
# plugins are configured by directory, so including it here preserves the
# existing access-manifest install path and does not replace that library.
add_subdirectory(
  "${CMAKE_CURRENT_LIST_DIR}/../layout_rewrite"
  "${CMAKE_CURRENT_BINARY_DIR}/layout_rewrite"
)

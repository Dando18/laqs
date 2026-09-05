add_library(LAQSTritonLayoutRewrite SHARED LayoutRewritePlugin.cpp)
add_dependencies(LAQSTritonLayoutRewrite
  TritonTableGen
  TritonGPUTableGen
)
target_link_libraries(LAQSTritonLayoutRewrite PRIVATE triton)
target_compile_options(LAQSTritonLayoutRewrite PRIVATE
  -fvisibility=hidden
  ${TRITON_DISABLE_EH_RTTI_FLAGS}
)

if(DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
  set_target_properties(LAQSTritonLayoutRewrite PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/../plugins"
  )
endif()

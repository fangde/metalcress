file(GLOB_RECURSE PROJECT_HEADER
    ${PROJECT_SOURCE_DIR}/include/core/*.h
    ${PROJECT_SOURCE_DIR}/include/core/*.cuh
    ${PROJECT_SOURCE_DIR}/include/math/*.h
    ${PROJECT_SOURCE_DIR}/include/math/*.cuh
    ${PROJECT_SOURCE_DIR}/include/utils/*.h
    ${PROJECT_SOURCE_DIR}/include/utils/*.cuh
    ${PROJECT_SOURCE_DIR}/include/solver/*.h
    ${PROJECT_SOURCE_DIR}/include/solver/*.cuh
    ${PROJECT_SOURCE_DIR}/include/engine/*.h
    ${PROJECT_SOURCE_DIR}/include/engine/*.cuh
    ${PROJECT_SOURCE_DIR}/include/geometry_utils/*.h
    ${PROJECT_SOURCE_DIR}/include/geometry_utils/*.cuh
)

file(GLOB_RECURSE INTERNAL_HEADER
    ${PROJECT_SOURCE_DIR}/src/core/*.h
    ${PROJECT_SOURCE_DIR}/src/core/*.cuh
    ${PROJECT_SOURCE_DIR}/src/solver/*.h
    ${PROJECT_SOURCE_DIR}/src/solver/*.cuh
    ${PROJECT_SOURCE_DIR}/src/engine/*.h
    ${PROJECT_SOURCE_DIR}/src/engine/*.cuh
    ${PROJECT_SOURCE_DIR}/src/utils/*.h
    ${PROJECT_SOURCE_DIR}/src/utils/*.cuh
)

if(APPLE)
    file(GLOB_RECURSE PROJECT_SOURCES
        ${PROJECT_SOURCE_DIR}/src/core/*.mm
        ${PROJECT_SOURCE_DIR}/src/core/*.cpp
        ${PROJECT_SOURCE_DIR}/src/solver/*.mm
        ${PROJECT_SOURCE_DIR}/src/solver/*.cpp
        ${PROJECT_SOURCE_DIR}/src/engine/*.mm
        ${PROJECT_SOURCE_DIR}/src/engine/*.cpp
        ${PROJECT_SOURCE_DIR}/src/geometry_utils/*.mm
        ${PROJECT_SOURCE_DIR}/src/geometry_utils/*.cpp
        ${PROJECT_SOURCE_DIR}/src/utils/*.mm
        ${PROJECT_SOURCE_DIR}/src/utils/*.cpp
    )
else()
    file(GLOB_RECURSE PROJECT_SOURCES
        ${PROJECT_SOURCE_DIR}/src/core/*.cu
        ${PROJECT_SOURCE_DIR}/src/core/*.cpp
        ${PROJECT_SOURCE_DIR}/src/solver/*.cu
        ${PROJECT_SOURCE_DIR}/src/solver/*.cpp
        ${PROJECT_SOURCE_DIR}/src/engine/*.cu
        ${PROJECT_SOURCE_DIR}/src/engine/*.cpp
        ${PROJECT_SOURCE_DIR}/src/geometry_utils/*.cu
        ${PROJECT_SOURCE_DIR}/src/geometry_utils/*.cpp
    )
endif()

if(ENGINE_STATIC)
    add_library(crmpm_engine STATIC ${PROJECT_SOURCES})
else()
    add_library(crmpm_engine SHARED ${PROJECT_SOURCES})
    add_compile_definitions(CRMPM_ENGINE_EXPORTS)
    set_target_properties(crmpm_engine PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
    )

    if(WIN32)
        target_sources(crmpm_engine PRIVATE ${CMAKE_BINARY_DIR}/generated/version.rc)
    endif()
endif()

set_target_properties(crmpm_engine PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_sources(crmpm_engine PRIVATE ${PROJECT_HEADER} ${INTERNAL_HEADER})

source_group(
    TREE "${PROJECT_SOURCE_DIR}/include"
    PREFIX "Header Files"
    FILES ${PROJECT_HEADER}
)

source_group(
    TREE "${PROJECT_SOURCE_DIR}/src"
    PREFIX "Header Files"
    FILES ${INTERNAL_HEADER}
)

source_group(
    TREE "${PROJECT_SOURCE_DIR}/src"
    PREFIX "Source Files"
    FILES ${PROJECT_SOURCES}
)

if(NOT APPLE)
    target_include_directories(crmpm_engine PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()
target_include_directories(crmpm_engine PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(crmpm_engine PUBLIC ${PROJECT_SOURCE_DIR}/include/core)
target_include_directories(crmpm_engine PUBLIC ${PROJECT_SOURCE_DIR}/include/math)
target_include_directories(crmpm_engine PUBLIC ${PROJECT_SOURCE_DIR}/include/solver)
target_include_directories(crmpm_engine PUBLIC ${PROJECT_SOURCE_DIR}/include/utils)
target_include_directories(crmpm_engine PUBLIC ${PROJECT_SOURCE_DIR}/include/engine)
target_include_directories(crmpm_engine PUBLIC ${PROJECT_SOURCE_DIR}/include/geometry_utils)

target_include_directories(crmpm_engine PRIVATE ${PROJECT_SOURCE_DIR}/src/core)
target_include_directories(crmpm_engine PRIVATE ${PROJECT_SOURCE_DIR}/src/solver)
target_include_directories(crmpm_engine PRIVATE ${PROJECT_SOURCE_DIR}/src/engine)
target_include_directories(crmpm_engine PRIVATE ${PROJECT_SOURCE_DIR}/src/utils)

target_include_directories(crmpm_engine PRIVATE ${PROJECT_SOURCE_DIR}/extern)

if(APPLE)
    target_link_libraries(crmpm_engine PRIVATE "-framework Metal" "-framework MetalKit" "-framework Foundation" "-framework Cocoa")
else()
    target_link_libraries(crmpm_engine PRIVATE cudart_static)
endif()

set_target_properties(crmpm_engine
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
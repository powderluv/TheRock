# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Shared by each build profile after all subprojects and artifacts are declared.
therock_subproject_merge_compile_commands()
therock_write_subproject_manifest()
include(therock_emit_consumer_graph)
therock_emit_consumer_graph("${CMAKE_BINARY_DIR}/therock_consumer_graph.json")

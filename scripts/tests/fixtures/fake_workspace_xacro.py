#!/usr/bin/env python3
import json
from pathlib import Path

from ament_index_python.packages import get_package_share_directory


package = Path(get_package_share_directory("included_description"))
dimensions = json.loads((package / "dimensions.json").read_text(encoding="utf-8"))
prefix = ""
for argument in __import__("sys").argv[2:]:
    if argument.startswith("prefix:="):
        prefix = argument.split(":=", 1)[1]

print(f'''<?xml version="1.0"?>
<!-- generated from package path {package} -->
<robot name="workspace_demo">
  <link name="{prefix}base">
    <collision><geometry><mesh filename="file://{package}/shape.stl"/></geometry></collision>
  </link>
  <link name="{prefix}tip">
    <visual><geometry><box size="{dimensions['x']} {dimensions['y']} {dimensions['z']}"/></geometry></visual>
  </link>
  <joint name="{prefix}mount" type="fixed">
    <parent link="{prefix}base"/><child link="{prefix}tip"/>
    <origin xyz="0 0 {dimensions['z']}" rpy="0 0 0"/>
  </joint>
</robot>''')

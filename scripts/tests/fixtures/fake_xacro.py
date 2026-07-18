#!/usr/bin/env python3
import sys

height = "0.1"
for argument in sys.argv[2:]:
    if argument.startswith("base_height:="):
        height = argument.split(":=", 1)[1]

print(f'''<?xml version="1.0"?>
<robot name="xacro_demo">
  <link name="base_link">
    <visual><geometry><box size="1 1 {height}"/></geometry></visual>
  </link>
</robot>''')

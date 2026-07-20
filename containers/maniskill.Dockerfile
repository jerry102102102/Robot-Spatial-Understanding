FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Keep this image aligned with the live PickCube evidence environment. MPlib in the
# ManiSkill 3.0.1 motion-planning example requires the NumPy 1.x ABI.

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . /workspace
RUN python3 -m pip install --no-cache-dir '.[maniskill]'

ENTRYPOINT ["robot-spatial"]
CMD ["--help"]

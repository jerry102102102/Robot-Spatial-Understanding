FROM python:3.11-slim

WORKDIR /workspace
COPY . /workspace
RUN python -m pip install --no-cache-dir .

ENTRYPOINT ["robot-spatial"]
CMD ["--help"]

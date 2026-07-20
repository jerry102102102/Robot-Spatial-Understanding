"""Typed user-facing errors for the simulation evidence package."""


class RobotSpatialUnderstandingError(ValueError):
    """Base error rendered by the public CLI without a traceback."""


class SchemaError(RobotSpatialUnderstandingError):
    """An artifact does not satisfy its declared contract."""


class IntegrityError(RobotSpatialUnderstandingError):
    """An artifact digest, path, or source binding is invalid."""


class EvidenceError(RobotSpatialUnderstandingError):
    """Evidence cannot be interpreted under the declared policy."""


class AdapterError(RobotSpatialUnderstandingError):
    """A simulator adapter rejected or could not normalize a source."""


class OracleIsolationError(RobotSpatialUnderstandingError):
    """Benchmark reference labels crossed the prediction boundary."""

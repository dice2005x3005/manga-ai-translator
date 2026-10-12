from pydantic import ValidationError


def format_validation_error(exc: ValidationError) -> str:
    """Render pydantic errors as one indented ``location: message`` line each."""
    lines = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "<root>"
        lines.append(f"  {location}: {error['msg']}")
    return "\n".join(lines)

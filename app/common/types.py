import annotated_types
from typing import Annotated

NonEmptyString = Annotated[str, annotated_types.MinLen(0)]
class MultipleOperationsFoundError(Exception):
    """
    Exception raised when multiple operations are found when looking for a specific one.

    This exception is usually raised when the user looks for a FeatureOperation instance,
    but too many attributes are left unspecified (set to None).
    """

    pass


class MultipleObjectsInFileError(Exception):
    """
    Exception raised when multiple objects are found in the same file.

    This exception is usually raised when the file read by "shelve" package contains
    multiple instances of DataFrameWithInfo.
    """

    pass


class NotShelveFileError(Exception):
    """
    Exception raised when the file is not importable by "shelve" package.

    Usually this happens when the file was not created by "shelve" package and
    "shelve" raises the message "no db type could be determined".
    """

    pass

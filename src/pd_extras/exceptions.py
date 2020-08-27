class MultipleOperationsFoundError(Exception):
    """
    Exception raised when multiple operations are found when looking for a specific one.

    This exception is usually raised when the user looks into a
    DataFrameWithInfo instance to find a specific FeatureOperation instance
    (stored in the "feature_elaborations" argument), but too many attributes are
    left unspecified (set to None).
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

    Usually this happens when the file was not created by "shelve" package.
    """

    pass

# encoding: utf-8

"""CSV source files loader for testing purposes."""

import os


class LazyResponder(object):
    """Loads and caches fixtures files by name from fixture directory.
    Provides access to all the svs fixtures in a directory by
    a standardized mapping of the file name, e.g. ca-1-c.csv is available
    as the `.CA_1_C` attribute of the loader.

    The fixture directory is specified relative to this (fixture root) directory.
    """

    def __init__(self, relpath):
        self._relpath = relpath
        self._cache = {}

    def __getattr__(self, fixture_name):
        if fixture_name not in self._cache:
            self._load_to_cache(fixture_name)
        return self._cache[fixture_name]

    @property
    def _dirpath(self):
        thisdir = os.path.dirname(os.path.abspath(__file__))
        return os.path.abspath(os.path.join(thisdir, self._relpath))


class LazyCSVResponseLoader(LazyResponder):
    """Specific class for CSV fixtures loader"""

    def _csv_path(self, fixture_name):
        return "%s/%s.csv" % (self._dirpath, fixture_name.replace("_", "-").lower())

    def _load_csv(self, path):
        with open(path, "rb") as f:
            csv_file = f.name
        return csv_file

    def _load_to_cache(self, fixture_name):
        csv_path = self._csv_path(fixture_name)
        if not os.path.exists(csv_path):
            raise ValueError("no CSV fixture found at %s" % csv_path)
        self._cache[fixture_name] = self._load_csv(csv_path)


CSV = LazyCSVResponseLoader("./csv")

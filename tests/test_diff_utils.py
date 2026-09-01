# pyright: reportPrivateUsage=false
"""Tests for shared diff-analysis helpers."""

from src.diff_utils import is_trivial_diff


class TestIsTrivialDiff:
    def test_whitespace_only_is_trivial(self):
        diff = "+   \n+\n-  \n- "
        assert is_trivial_diff(diff) is True

    def test_comment_lines_are_trivial(self):
        diff = (
            "+ # add a comment\n- // old comment\n+ /* block comment */"
            "\n+ * middle\n+ -- sql comment"
        )
        assert is_trivial_diff(diff) is True

    def test_mixed_whitespace_and_comments_is_trivial(self):
        diff = "+\n+ # comment\n-   \n- // removed comment"
        assert is_trivial_diff(diff) is True

    def test_real_code_is_not_trivial(self):
        diff = "+ def new_function():\n+     return True"
        assert is_trivial_diff(diff) is False

    def test_import_is_not_trivial(self):
        diff = "+ import os"
        assert is_trivial_diff(diff) is False

    def test_python_from_import_is_not_trivial(self):
        diff = "+ from .signals import *"
        assert is_trivial_diff(diff) is False

    def test_js_require_is_not_trivial(self):
        diff = '+ const x = require("./polyfills")'
        assert is_trivial_diff(diff) is False

    def test_php_include_is_not_trivial(self):
        diff = '+ include "bootstrap.php";'
        assert is_trivial_diff(diff) is False

    def test_cpp_include_is_not_trivial(self):
        diff = '+ #include <stdio.h>'
        assert is_trivial_diff(diff) is False

    def test_empty_diff_is_trivial(self):
        assert is_trivial_diff("") is True

    def test_context_lines_ignored(self):
        diff = " unchanged line\n+ # just a comment\n  another context"
        assert is_trivial_diff(diff) is True

    def test_mixed_trivial_and_code_is_not_trivial(self):
        diff = "+ # comment\n+ real_code = True"
        assert is_trivial_diff(diff) is False

    def test_triple_slash_comment_is_trivial(self):
        diff = "+ /// doc comment\n- /// old doc"
        assert is_trivial_diff(diff) is True

    def test_javadoc_comment_is_trivial(self):
        diff = "+ /** start\n+  * middle\n+  */ end"
        assert is_trivial_diff(diff) is True

    def test_use_statement_is_not_trivial(self):
        diff = "+ use App\\Models\\User;"
        assert is_trivial_diff(diff) is False

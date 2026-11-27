import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from multiprocessing.pool import Pool

from mars_rec.utils.utils import (
    chunks,
    literal_eval_if_str,
    parallel_literal_eval,
    clean_filename,
    reduce_df_mem,
    mean_confidence_interval,
    random_date,
)


class TestChunks:
    def test_basic_chunking(self):
        result = list(chunks([1, 2, 3, 4, 5, 6], 2))
        assert result == [[1, 2], [3, 4], [5, 6]]

    def test_uneven_chunks(self):
        result = list(chunks([1, 2, 3, 4, 5], 2))
        assert len(result) == 3
        assert result[-1] == [5]

    def test_chunk_size_larger_than_list(self):
        result = list(chunks([1, 2, 3], 10))
        assert result == [[1, 2, 3]]

    def test_empty_list(self):
        result = list(chunks([], 2))
        assert result == []

    def test_range_input(self):
        result = list(chunks(range(6), 3))
        assert len(result) == 2

    def test_chunk_size_one(self):
        result = list(chunks([1, 2, 3], 1))
        assert result == [[1], [2], [3]]


class TestLiteralEvalIfStr:
    def test_string_list(self):
        result = literal_eval_if_str("[1, 2, 3]")
        assert result == [1, 2, 3]

    def test_string_dict(self):
        result = literal_eval_if_str("{'a': 1}")
        assert result == {"a": 1}

    def test_non_string_passthrough(self):
        assert literal_eval_if_str(42) == 42
        assert literal_eval_if_str([1, 2]) == [1, 2]
        assert literal_eval_if_str(None) is None

    def test_string_number(self):
        result = literal_eval_if_str("3.14")
        assert result == pytest.approx(3.14)

    def test_nested_list(self):
        result = literal_eval_if_str("[[1, 2], [3, 4]]")
        assert result == [[1, 2], [3, 4]]


class TestCleanFilename:
    def test_spaces_replaced(self):
        result = clean_filename("my file name")
        assert " " not in result

    def test_special_chars_removed(self):
        result = clean_filename("file@name#test!")
        assert "@" not in result
        assert "#" not in result
        assert "!" not in result

    def test_alphanumeric_preserved(self):
        result = clean_filename("file123")
        assert "file123" in result

    def test_long_filename_truncated(self):
        long_name = "a" * 300
        result = clean_filename(long_name)
        assert len(result) <= 255

    def test_unicode_normalized(self):
        result = clean_filename("café")
        assert isinstance(result, str)

    def test_dashes_preserved(self):
        result = clean_filename("my-file-name")
        assert "my-file-name" == result

    def test_underscores_preserved(self):
        result = clean_filename("my_file_name")
        assert "my_file_name" == result


class TestReduceDfMem:
    def test_int_columns_downcast(self):
        df = pd.DataFrame({"small_int": [1, 2, 3], "large_float": [1.0, 2.0, 3.0]})
        result = reduce_df_mem(df)
        # Should be reduced from default int64/float64
        assert result["small_int"].dtype.itemsize <= 8
        assert result["large_float"].dtype.itemsize <= 8

    def test_returns_dataframe(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = reduce_df_mem(df)
        assert isinstance(result, pd.DataFrame)

    def test_object_columns_unchanged(self):
        df = pd.DataFrame({"text": ["a", "b", "c"], "num": [1, 2, 3]})
        result = reduce_df_mem(df)
        # pandas >= 2 may return StringDtype instead of object for string columns
        assert isinstance(result["text"].iloc[0], str)

    def test_large_int_stays_int64(self):
        large_val = np.iinfo(np.int64).max - 1
        df = pd.DataFrame({"big_int": [large_val]})
        result = reduce_df_mem(df)
        assert result["big_int"].dtype == np.int64

    def test_small_int_fits_in_int8(self):
        df = pd.DataFrame({"tiny": [1, 2, 3]})
        result = reduce_df_mem(df)
        assert result["tiny"].dtype == np.int8


class TestMeanConfidenceInterval:
    def test_returns_mean_and_interval(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        m, h = mean_confidence_interval(data)
        assert m == pytest.approx(3.0)
        assert h > 0

    def test_single_value(self):
        data = [5.0]
        m, h = mean_confidence_interval(data)
        assert m == pytest.approx(5.0)

    def test_nan_values_excluded(self):
        data = [1.0, float("nan"), 3.0]
        m, h = mean_confidence_interval(data)
        assert m == pytest.approx(2.0)

    def test_custom_confidence(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        m_95, h_95 = mean_confidence_interval(data, confidence=0.95)
        m_99, h_99 = mean_confidence_interval(data, confidence=0.99)
        assert h_99 > h_95


class TestRandomDate:
    def test_returns_generator(self):
        start = datetime(2020, 1, 1)
        gen = random_date(start, 5)
        dates = list(gen)
        assert len(dates) == 5

    def test_dates_after_start(self):
        start = datetime(2020, 1, 1)
        dates = list(random_date(start, 10))
        assert all(d >= start for d in dates)

    def test_zero_dates(self):
        start = datetime(2020, 1, 1)
        dates = list(random_date(start, 0))
        assert dates == []

    def test_dates_are_datetime(self):
        start = datetime(2020, 1, 1)
        dates = list(random_date(start, 3))
        assert all(isinstance(d, datetime) for d in dates)


class TestParallelLiteralEval:
    def test_parses_string_lists(self):
        series = pd.Series(["[1, 2, 3]", "[4, 5]", "[6]"])
        result = parallel_literal_eval(series, use_tqdm=False)
        assert result == [[1, 2, 3], [4, 5], [6]]

    def test_parses_string_floats(self):
        series = pd.Series(["[1.0, 2.5]", "[3.14]"])
        result = parallel_literal_eval(series, use_tqdm=False)
        assert result[0] == [1.0, 2.5]
        assert result[1] == [3.14]

    def test_preserves_non_string_elements(self):
        series = pd.Series([[1, 2], [3, 4]])
        result = parallel_literal_eval(series, use_tqdm=False)
        assert result == [[1, 2], [3, 4]]

    def test_empty_series(self):
        series = pd.Series([], dtype=object)
        result = parallel_literal_eval(series, use_tqdm=False)
        assert result == []

    def test_single_element(self):
        series = pd.Series(["[10, 20, 30]"])
        result = parallel_literal_eval(series, use_tqdm=False)
        assert result == [[10, 20, 30]]

    def test_with_shared_pool(self):
        series = pd.Series(["[1, 2]", "[3, 4]"])
        with Pool(2) as pool:
            result = parallel_literal_eval(series, pool=pool, use_tqdm=False)
        assert result == [[1, 2], [3, 4]]

    def test_mixed_string_and_non_string(self):
        series = pd.Series(["[1, 2]", [3, 4], "[5]"])
        result = parallel_literal_eval(series, use_tqdm=False)
        assert result[0] == [1, 2]
        assert result[1] == [3, 4]
        assert result[2] == [5]

    def test_nested_lists(self):
        series = pd.Series(["[[1, 2], [3, 4]]"])
        result = parallel_literal_eval(series, use_tqdm=False)
        assert result == [[[1, 2], [3, 4]]]


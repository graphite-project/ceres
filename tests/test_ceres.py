from unittest import TestCase
from mock import Mock, patch

from ceres import *


class ModuleFunctionsTest(TestCase):
  @patch('ceres.isdir', new=Mock(return_value=False))
  @patch('ceres.CeresTree', new=Mock(spec=CeresTree))
  def test_get_tree_with_no_tree(self):
    tree = getTree('/graphite/storage/ceres/foo/bar')
    self.assertEqual(None, tree)

  @patch('ceres.CeresTree', spec=CeresTree)
  @patch('ceres.isdir')
  def test_get_tree_with_tree_samedir(self, isdir_mock, ceres_tree_mock):
    isdir_mock.return_value = True
    tree = getTree('/graphite/storage/ceres')
    self.assertNotEqual(None, tree)
    isdir_mock.assert_called_once_with('/graphite/storage/ceres/.ceres-tree')
    ceres_tree_mock.assert_called_once_with('/graphite/storage/ceres')


class TimeSeriesDataTest(TestCase):
  def setUp(self):
    self.time_series = TimeSeriesData(0, 50, 5, [float(x) for x in xrange(0, 10)])

  def test_timestamps_property(self):
    self.assertEqual(10, len(self.time_series.timestamps))
    self.assertEqual(0, self.time_series.timestamps[0])
    self.assertEqual(45, self.time_series.timestamps[-1])

  def test_iter_values(self):
    values = list(self.time_series)
    self.assertEqual(10, len(values))
    self.assertEqual((0, 0.0), values[0])
    self.assertEqual((45, 9.0), values[-1])

  def test_merge_no_missing(self):
    # merge only has effect if time series has no gaps
    other_series = TimeSeriesData(0, 25, 5, [float(x * x) for x in xrange(1, 6)])
    original_values = list(self.time_series)
    self.time_series.merge(other_series)
    self.assertEqual(original_values, list(self.time_series))

  def test_merge_with_empty(self):
    new_series = TimeSeriesData(0, 50, 5, [None] * 10)
    new_series.merge(self.time_series)
    self.assertEqual(list(self.time_series), list(new_series))

  def test_merge_with_holes(self):
    values = []
    for x in xrange(0, 10):
      if x % 2 == 0:
        values.append(x)
      else:
        values.append(None)
    new_series = TimeSeriesData(0, 50, 5, values)
    new_series.merge(self.time_series)
    self.assertEqual(list(self.time_series), list(new_series))

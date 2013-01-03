from unittest import TestCase
from mock import ANY, Mock, mock_open, patch

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


class CeresTreeTest(TestCase):
  def setUp(self):
    with patch('ceres.isdir', new=Mock(return_value=True)):
      self.ceres_tree = CeresTree('/graphite/storage/ceres')


  @patch('ceres.isdir', new=Mock(return_value=False))
  def test_init_invalid(self):
    self.assertRaises(ValueError, CeresTree, '/nonexistent_path')

  @patch('ceres.isdir', new=Mock(return_value=True))
  @patch('ceres.abspath')
  def test_init_valid(self, abspath_mock):
    abspath_mock.return_value = '/var/graphite/storage/ceres'
    tree = CeresTree('/graphite/storage/ceres')
    abspath_mock.assert_called_once_with('/graphite/storage/ceres')
    self.assertEqual('/var/graphite/storage/ceres', tree.root)

  @patch('ceres.isdir', new=Mock(return_value=False))
  @patch.object(CeresTree, '__init__')
  @patch('os.makedirs')
  def test_create_tree_new_dir(self, makedirs_mock, ceres_tree_init_mock):
    ceres_tree_init_mock.return_value = None
    with patch('__builtin__.open', mock_open()) as open_mock:
      CeresTree.createTree('/graphite/storage/ceres')
      makedirs_mock.assert_called_once_with('/graphite/storage/ceres/.ceres-tree', DIR_PERMS)
      self.assertFalse(open_mock.called)
      ceres_tree_init_mock.assert_called_once_with('/graphite/storage/ceres')

  @patch('ceres.isdir', new=Mock(return_value=True))
  @patch.object(CeresTree, '__init__')
  @patch('os.makedirs')
  def test_create_tree_existing_dir(self, makedirs_mock, ceres_tree_init_mock):
    ceres_tree_init_mock.return_value = None
    with patch('__builtin__.open', mock_open()) as open_mock:
      CeresTree.createTree('/graphite/storage/ceres')
      self.assertFalse(makedirs_mock.called)
      self.assertFalse(open_mock.called)
      ceres_tree_init_mock.assert_called_once_with('/graphite/storage/ceres')

  @patch('ceres.isdir', new=Mock(return_value=True))
  @patch.object(CeresTree, '__init__', new=Mock(return_value=None))
  @patch('os.makedirs', new=Mock())
  def test_create_tree_write_props(self):
    props = {
      "foo_prop": "foo_value",
      "bar_prop": "bar_value"}
    with patch('__builtin__.open', mock_open()) as open_mock:
      CeresTree.createTree('/graphite/storage/ceres', **props)
      for (prop,value) in props.items():
        open_mock.assert_any_call(join('/graphite/storage/ceres', '.ceres-tree', prop), 'w')
        open_mock.return_value.write.assert_any_call(value)

  @patch('ceres.abspath', new=Mock(side_effect=lambda x: x))
  def test_get_node_path_clean(self):
    result = self.ceres_tree.getNodePath('/graphite/storage/ceres/metric/foo')
    self.assertEqual('metric.foo', result)

  @patch('ceres.abspath', new=Mock(side_effect=lambda x: x))
  def test_get_node_path_trailing_slash(self):
    result = self.ceres_tree.getNodePath('/graphite/storage/ceres/metric/foo/')
    self.assertEqual('metric.foo', result)

  @patch('ceres.abspath', new=Mock(side_effect=lambda x: x))
  def test_get_node_path_outside_tree(self):
    self.assertRaises(ValueError, self.ceres_tree.getNodePath, '/metric/foo')

  @patch('ceres.CeresNode', spec=CeresNode)
  def test_get_node_uncached(self, ceres_node_mock):
    ceres_node_mock.isNodeDir.return_value = True
    result = self.ceres_tree.getNode('metrics.foo')
    ceres_node_mock.assert_called_once_with(
      self.ceres_tree,
      'metrics.foo',
      '/graphite/storage/ceres/metrics/foo')
    self.assertEqual(result, ceres_node_mock())

  @patch('ceres.CeresNode', spec=CeresNode)
  @patch('ceres.abspath', new=Mock(side_effect=lambda x: x))
  @patch('ceres.glob', new=Mock(side_effect=lambda x: [x]))
  def test_find_explicit_metric(self, ceres_node_mock):
    ceres_node_mock.isNodeDir.return_value = True
    result = list(self.ceres_tree.find('metrics.foo'))
    self.assertEqual(1, len(result))
    self.assertEqual(result[0], ceres_node_mock())

  @patch('ceres.CeresNode', spec=CeresNode)
  @patch('ceres.abspath', new=Mock(side_effect=lambda x: x))
  @patch('ceres.glob')
  def test_find_wildcard(self, glob_mock, ceres_node_mock):
    matches = ['foo', 'bar', 'baz']
    glob_mock.side_effect = lambda x: [x.replace('*', m) for m in matches]
    ceres_node_mock.isNodeDir.return_value = True
    result = list(self.ceres_tree.find('metrics.*'))
    self.assertEqual(3, len(result))
    ceres_node_mock.assert_any_call(self.ceres_tree, 'metrics.foo', ANY)
    ceres_node_mock.assert_any_call(self.ceres_tree, 'metrics.bar', ANY)
    ceres_node_mock.assert_any_call(self.ceres_tree, 'metrics.baz', ANY)

  @patch('ceres.CeresNode', spec=CeresNode)
  @patch('ceres.abspath', new=Mock(side_effect=lambda x: x))
  @patch('ceres.glob', new=Mock(return_value=[]))
  def test_find_wildcard_no_matches(self, ceres_node_mock):
    ceres_node_mock.isNodeDir.return_value = False
    result = list(self.ceres_tree.find('metrics.*'))
    self.assertEqual(0, len(result))
    self.assertFalse(ceres_node_mock.called)

  @patch('ceres.CeresNode', spec=CeresNode)
  @patch('ceres.abspath', new=Mock(side_effect=lambda x: x))
  @patch('ceres.glob', new=Mock(side_effect=lambda x: [x]))
  def test_find_metric_with_interval(self, ceres_node_mock):
    ceres_node_mock.isNodeDir.return_value = True
    ceres_node_mock.return_value.hasDataForInterval.return_value = False
    result = list(self.ceres_tree.find('metrics.foo', 0, 1000))
    self.assertEqual(0, len(result))
    ceres_node_mock.return_value.hasDataForInterval.assert_called_once_with(0, 1000)

  @patch('ceres.CeresNode', spec=CeresNode)
  @patch('ceres.abspath', new=Mock(side_effect=lambda x: x))
  @patch('ceres.glob', new=Mock(side_effect=lambda x: [x]))
  def test_find_metric_with_interval_not_found(self, ceres_node_mock):
    ceres_node_mock.isNodeDir.return_value = True
    ceres_node_mock.return_value.hasDataForInterval.return_value = True
    result = list(self.ceres_tree.find('metrics.foo', 0, 1000))
    self.assertEqual(result[0], ceres_node_mock())
    ceres_node_mock.return_value.hasDataForInterval.assert_called_once_with(0, 1000)

  def test_store_invalid_node(self):
    with patch.object(self.ceres_tree, 'getNode', new=Mock(return_value=None)):
      datapoints = [(100, 1.0)]
      self.assertRaises(NodeNotFound, self.ceres_tree.store, 'metrics.foo', datapoints)

  @patch('ceres.CeresNode', spec=CeresNode)
  def test_store_valid_node(self, ceres_node_mock):
    datapoints = [(100, 1.0)]
    self.ceres_tree.store('metrics.foo', datapoints)
    ceres_node_mock.assert_called_once_with(self.ceres_tree, 'metrics.foo', ANY)
    ceres_node_mock.return_value.write.assert_called_once_with(datapoints)

  def fetch_invalid_node(self):
    with patch.object(self.ceres_tree, 'getNode', new=Mock(return_value=None)):
      self.assertRaises(NodeNotFound, self.ceres_tree.fetch, 'metrics.foo')

  @patch('ceres.CeresNode', spec=CeresNode)
  def fetch_metric(self, ceres_node_mock):
    read_mock = ceres_node_mock.return_value.read
    read_mock.return_value = Mock(spec=TimeSeriesData)
    result = self.ceres_tree.fetch('metrics.foo', 0, 1000)
    ceres_node_mock.assert_called_once_with(self.ceres_tree, 'metrics.foo', ANY)
    read_mock.assert_called_once_with(0, 1000)
    self.assertEqual(Mock(spec=TimeSeriesData), result)

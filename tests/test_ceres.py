from unittest import TestCase
from mock import ANY, Mock, mock_open, patch

from ceres import *


def fetch_mock_open_writes(open_mock):
  handle = open_mock()
  return ''.join([ c[0][0] for c in handle.write.call_args_list])

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


class CeresNodeTest(TestCase):
  def setUp(self):
    with patch('ceres.isdir', new=Mock(return_value=True)):
      with patch('ceres.exists', new=Mock(return_value=True)):
        self.ceres_tree = CeresTree('/graphite/storage/ceres')
        self.ceres_node = CeresNode(self.ceres_tree, 'sample_metric', '/graphite/storage/ceres/sample_metric')

  def test_init_sets_default_cache_behavior(self):
    ceres_node = CeresNode(self.ceres_tree, 'sample_metric', '/graphite/storage/ceres/sample_metric')
    self.assertEqual(DEFAULT_SLICE_CACHING_BEHAVIOR, ceres_node.sliceCachingBehavior)

  @patch('ceres.os.makedirs', new=Mock())
  @patch('ceres.CeresNode.writeMetadata')
  def test_create_sets_a_default_timestep(self, write_metadata_mock):
    ceres_node = CeresNode.create(self.ceres_tree, 'sample_metric')
    write_metadata_mock.assert_called_with(dict(timeStep=DEFAULT_TIMESTEP))

  @patch('ceres.os.makedirs', new=Mock())
  @patch('ceres.CeresNode.writeMetadata', new=Mock())
  def test_create_returns_new_ceres_node(self):
    ceres_node = CeresNode.create(self.ceres_tree, 'sample_metric')
    self.assertTrue(isinstance(ceres_node, CeresNode))

  def test_write_metadata(self):
    import json

    open_mock = mock_open()
    metadata = dict(timeStep=60, aggregationMethod='avg')
    with patch('__builtin__.open', open_mock):
      self.ceres_node.writeMetadata(metadata)
    self.assertEquals(json.dumps(metadata), fetch_mock_open_writes(open_mock))

  def test_read_metadata_sets_timestep(self):
    import json

    metadata = dict(timeStep=60, aggregationMethod='avg')
    json_metadata = json.dumps(metadata)
    open_mock = mock_open(read_data=json_metadata)
    with patch('__builtin__.open', open_mock):
      self.ceres_node.readMetadata()
    open_mock().read.assert_called_once()
    self.assertEqual(60, self.ceres_node.timeStep)

  def test_set_slice_caching_behavior_validates_names(self):
    self.ceres_node.setSliceCachingBehavior('none')
    self.assertEquals('none', self.ceres_node.sliceCachingBehavior)
    self.ceres_node.setSliceCachingBehavior('all')
    self.assertEquals('all', self.ceres_node.sliceCachingBehavior)
    self.ceres_node.setSliceCachingBehavior('latest')
    self.assertEquals('latest', self.ceres_node.sliceCachingBehavior)
    self.assertRaises(ValueError, self.ceres_node.setSliceCachingBehavior, 'foo')
    # Assert unchanged
    self.assertEquals('latest', self.ceres_node.sliceCachingBehavior)

  def test_slices_is_a_generator(self):
    from types import GeneratorType

    self.assertTrue(isinstance(self.ceres_node.slices, GeneratorType))

  def test_slices_returns_cached_set_when_behavior_is_all(self):
    def mock_slice():
      return Mock(spec=CeresSlice)

    self.ceres_node.setSliceCachingBehavior('all')
    cached_contents = [ mock_slice for c in range(4) ]
    self.ceres_node.sliceCache = cached_contents
    with patch('ceres.CeresNode.readSlices') as read_slices_mock:
      slice_list = list(self.ceres_node.slices)
      self.assertFalse(read_slices_mock.called)

    self.assertEquals(cached_contents, slice_list)

  def test_slices_returns_first_cached_when_behavior_is_latest(self):
    self.ceres_node.setSliceCachingBehavior('latest')
    cached_contents = Mock(spec=CeresSlice)
    self.ceres_node.sliceCache = cached_contents

    read_slices_mock = Mock(return_value=[])
    with patch('ceres.CeresNode.readSlices', new=read_slices_mock):
      slice_iter = self.ceres_node.slices
      self.assertEquals(cached_contents, slice_iter.next())
      # We should be yielding cached before trying to read
      self.assertFalse(read_slices_mock.called)

  def test_slices_reads_remaining_when_behavior_is_latest(self):
    self.ceres_node.setSliceCachingBehavior('latest')
    cached_contents = Mock(spec=CeresSlice)
    self.ceres_node.sliceCache = cached_contents

    read_slices_mock = Mock(return_value=[(0,60)])
    with patch('ceres.CeresNode.readSlices', new=read_slices_mock):
      slice_iter = self.ceres_node.slices
      slice_iter.next()

      # *now* we expect to read from disk
      try:
        while True:
          slice_iter.next()
      except StopIteration:
        pass

    read_slices_mock.assert_called_once_with()

  def test_slices_reads_from_disk_when_behavior_is_none(self):
    self.ceres_node.setSliceCachingBehavior('none')
    read_slices_mock = Mock(return_value=[(0,60)])
    with patch('ceres.CeresNode.readSlices', new=read_slices_mock):
      slice_iter = self.ceres_node.slices
      slice_iter.next()

    read_slices_mock.assert_called_once_with()

  def test_slices_reads_from_disk_when_cache_empty_and_behavior_all(self):
    self.ceres_node.setSliceCachingBehavior('all')
    read_slices_mock = Mock(return_value=[(0,60)])
    with patch('ceres.CeresNode.readSlices', new=read_slices_mock):
      slice_iter = self.ceres_node.slices
      slice_iter.next()

    read_slices_mock.assert_called_once_with()

  def test_slices_reads_from_disk_when_cache_empty_and_behavior_latest(self):
    self.ceres_node.setSliceCachingBehavior('all')
    read_slices_mock = Mock(return_value=[(0,60)])
    with patch('ceres.CeresNode.readSlices', new=read_slices_mock):
      slice_iter = self.ceres_node.slices
      slice_iter.next()

    read_slices_mock.assert_called_once_with()

  @patch('ceres.exists', new=Mock(return_value=False))
  def test_read_slices_raises_when_node_doesnt_exist(self):
    self.assertRaises(NodeDeleted, self.ceres_node.readSlices)

  @patch('ceres.exists', new=Mock(return_Value=True))
  def test_read_slices_ignores_not_slices(self):
    listdir_mock = Mock(return_value=['0@60.slice', '0@300.slice', 'foo'])
    with patch('ceres.os.listdir', new=listdir_mock):
      self.assertEquals(2, len(self.ceres_node.readSlices()))


class CeresSliceTest(TestCase):
  def setUp(self):
    with patch('ceres.isdir', new=Mock(return_value=True)):
      with patch('ceres.exists', new=Mock(return_value=True)):
        self.ceres_tree = CeresTree('/graphite/storage/ceres')
        self.ceres_node = CeresNode(self.ceres_tree, 'sample_metric', '/graphite/storage/ceres/sample_metric')

  def test_init_sets_fspath_name(self):
    ceres_slice = CeresSlice(self.ceres_node, 0, 60)
    self.assertTrue(ceres_slice.fsPath.endswith('0@60.slice'))



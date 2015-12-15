from unittest import TestCase, skip

import errno

from mock import ANY, Mock, call, mock_open, patch
from os import path

try:
  import __builtin__ as builtins
except ImportError:
  import builtins


from ceres import CeresNode, CeresSlice, CeresTree
from ceres import DATAPOINT_SIZE, DEFAULT_SLICE_CACHING_BEHAVIOR, DEFAULT_TIMESTEP, DIR_PERMS,\
    MAX_SLICE_GAP
from ceres import getTree, CorruptNode, NoData, NodeDeleted, NodeNotFound, SliceDeleted,\
    SliceGapTooLarge, TimeSeriesData


def fetch_mock_open_writes(open_mock):
  handle = open_mock()
  # XXX Python3 compability since a write can be bytes or str
  try:
    return b''.join([c[0][0] for c in handle.write.call_args_list])
  except TypeError:
    return ''.join([c[0][0] for c in handle.write.call_args_list])


def make_slice_mock(start, end, step):
  slice_mock = Mock(spec=CeresSlice)
  slice_mock.startTime = start
  slice_mock.endTime = end
  slice_mock.timeStep = step

  def side_effect(*args, **kwargs):
    startTime, endTime = args
    result_start = max(startTime, start)
    result_end = min(endTime, end)
    points = (result_end - result_start) // step
    return TimeSeriesData(result_start, result_end, step, [0] * points)

  slice_mock.read.side_effect = side_effect
  return slice_mock


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
    self.time_series = TimeSeriesData(0, 50, 5, [float(x) for x in range(0, 10)])

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
    other_series = TimeSeriesData(0, 25, 5, [float(x * x) for x in range(1, 6)])
    original_values = list(self.time_series)
    self.time_series.merge(other_series)
    self.assertEqual(original_values, list(self.time_series))

  def test_merge_with_empty(self):
    new_series = TimeSeriesData(0, 50, 5, [None] * 10)
    new_series.merge(self.time_series)
    self.assertEqual(list(self.time_series), list(new_series))

  def test_merge_with_holes(self):
    values = []
    for x in range(0, 10):
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
    with patch.object(builtins, 'open', mock_open()) as open_mock:
      CeresTree.createTree('/graphite/storage/ceres')
      makedirs_mock.assert_called_once_with('/graphite/storage/ceres/.ceres-tree', DIR_PERMS)
      self.assertFalse(open_mock.called)
      ceres_tree_init_mock.assert_called_once_with('/graphite/storage/ceres')

  @patch('ceres.isdir', new=Mock(return_value=True))
  @patch.object(CeresTree, '__init__')
  @patch('os.makedirs')
  def test_create_tree_existing_dir(self, makedirs_mock, ceres_tree_init_mock):
    ceres_tree_init_mock.return_value = None
    with patch.object(builtins, 'open', mock_open()) as open_mock:
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
    with patch.object(builtins, 'open', mock_open()) as open_mock:
      CeresTree.createTree('/graphite/storage/ceres', **props)
      for (prop, value) in props.items():
        open_mock.assert_any_call(path.join('/graphite/storage/ceres', '.ceres-tree', prop), 'w')
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
        self.ceres_node = CeresNode(
            self.ceres_tree,
            'sample_metric',
            '/graphite/storage/ceres/sample_metric')
        self.ceres_node.timeStep = 60

    slice_configs = [
      (1200, 1800, 60),
      (600, 1200, 60)]

    self.ceres_slices = []
    for start, end, step in slice_configs:
      slice_mock = make_slice_mock(start, end, step)
      self.ceres_slices.append(slice_mock)

  def test_init_sets_default_cache_behavior(self):
    ceres_node = CeresNode(
        self.ceres_tree,
        'sample_metric',
        '/graphite/storage/ceres/sample_metric')
    self.assertEqual(DEFAULT_SLICE_CACHING_BEHAVIOR, ceres_node.sliceCachingBehavior)

  @patch('ceres.os.makedirs', new=Mock())
  @patch('ceres.CeresNode.writeMetadata')
  def test_create_sets_a_default_timestep(self, write_metadata_mock):
    CeresNode.create(self.ceres_tree, 'sample_metric')
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
    with patch.object(builtins, 'open', open_mock):
      self.ceres_node.writeMetadata(metadata)
    self.assertEquals(json.dumps(metadata), fetch_mock_open_writes(open_mock))

  def test_read_metadata_sets_timestep(self):
    import json

    metadata = dict(timeStep=60, aggregationMethod='avg')
    json_metadata = json.dumps(metadata)
    open_mock = mock_open(read_data=json_metadata)
    with patch.object(builtins, 'open', open_mock):
      self.ceres_node.readMetadata()
    open_mock().read.assert_called_once()
    self.assertEqual(60, self.ceres_node.timeStep)

  def test_read_metadata_returns_corrupt_if_json_error(self):
    with patch.object(builtins, 'open', mock_open()):
      self.assertRaises(CorruptNode, self.ceres_node.readMetadata)

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
    cached_contents = [mock_slice for c in range(4)]
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
      self.assertEquals(cached_contents, next(slice_iter))
      # We should be yielding cached before trying to read
      self.assertFalse(read_slices_mock.called)

  def test_slices_reads_remaining_when_behavior_is_latest(self):
    self.ceres_node.setSliceCachingBehavior('latest')
    cached_contents = Mock(spec=CeresSlice)
    self.ceres_node.sliceCache = cached_contents

    read_slices_mock = Mock(return_value=[(0, 60)])
    with patch('ceres.CeresNode.readSlices', new=read_slices_mock):
      slice_iter = self.ceres_node.slices
      next(slice_iter)

      # *now* we expect to read from disk
      try:
        while True:
          next(slice_iter)
      except StopIteration:
        pass

    read_slices_mock.assert_called_once_with()

  def test_slices_reads_from_disk_when_behavior_is_none(self):
    self.ceres_node.setSliceCachingBehavior('none')
    read_slices_mock = Mock(return_value=[(0, 60)])
    with patch('ceres.CeresNode.readSlices', new=read_slices_mock):
      slice_iter = self.ceres_node.slices
      next(slice_iter)

    read_slices_mock.assert_called_once_with()

  def test_slices_reads_from_disk_when_cache_empty_and_behavior_all(self):
    self.ceres_node.setSliceCachingBehavior('all')
    read_slices_mock = Mock(return_value=[(0, 60)])
    with patch('ceres.CeresNode.readSlices', new=read_slices_mock):
      slice_iter = self.ceres_node.slices
      next(slice_iter)

    read_slices_mock.assert_called_once_with()

  def test_slices_reads_from_disk_when_cache_empty_and_behavior_latest(self):
    self.ceres_node.setSliceCachingBehavior('all')
    read_slices_mock = Mock(return_value=[(0, 60)])
    with patch('ceres.CeresNode.readSlices', new=read_slices_mock):
      slice_iter = self.ceres_node.slices
      next(slice_iter)

    read_slices_mock.assert_called_once_with()

  @patch('ceres.exists', new=Mock(return_value=False))
  def test_read_slices_raises_when_node_doesnt_exist(self):
    self.assertRaises(NodeDeleted, self.ceres_node.readSlices)

  @patch('ceres.exists', new=Mock(return_Value=True))
  def test_read_slices_ignores_not_slices(self):
    listdir_mock = Mock(return_value=['0@60.slice', '0@300.slice', 'foo'])
    with patch('ceres.os.listdir', new=listdir_mock):
      self.assertEquals(2, len(self.ceres_node.readSlices()))

  @patch('ceres.exists', new=Mock(return_Value=True))
  def test_read_slices_parses_slice_filenames(self):
    listdir_mock = Mock(return_value=['0@60.slice', '0@300.slice'])
    with patch('ceres.os.listdir', new=listdir_mock):
      slice_infos = self.ceres_node.readSlices()
      self.assertTrue((0, 60) in slice_infos)
      self.assertTrue((0, 300) in slice_infos)

  @patch('ceres.exists', new=Mock(return_Value=True))
  def test_read_slices_reverse_sorts_by_time(self):
    listdir_mock = Mock(return_value=[
      '0@60.slice',
      '320@300.slice',
      '120@120.slice',
      '0@120.slice',
      '600@300.slice'])

    with patch('ceres.os.listdir', new=listdir_mock):
      slice_infos = self.ceres_node.readSlices()
      slice_timestamps = [s[0] for s in slice_infos]
      self.assertEqual([600, 320, 120, 0, 0], slice_timestamps)

  def test_no_data_exists_if_no_slices_exist(self):
    with patch('ceres.CeresNode.readSlices', new=Mock(return_value=[])):
      self.assertFalse(self.ceres_node.hasDataForInterval(0, 60))

  def test_no_data_exists_if_no_slices_exist_and_no_time_specified(self):
    with patch('ceres.CeresNode.readSlices', new=Mock(return_value=[])):
      self.assertFalse(self.ceres_node.hasDataForInterval(None, None))

  def test_data_exists_if_slices_exist_and_no_time_specified(self):
    with patch('ceres.CeresNode.slices', new=self.ceres_slices):
      self.assertTrue(self.ceres_node.hasDataForInterval(None, None))

  def test_data_exists_if_slice_covers_interval_completely(self):
    with patch('ceres.CeresNode.slices', new=[self.ceres_slices[0]]):
      self.assertTrue(self.ceres_node.hasDataForInterval(1200, 1800))

  def test_data_exists_if_slice_covers_interval_end(self):
    with patch('ceres.CeresNode.slices', new=[self.ceres_slices[0]]):
      self.assertTrue(self.ceres_node.hasDataForInterval(600, 1260))

  def test_data_exists_if_slice_covers_interval_start(self):
    with patch('ceres.CeresNode.slices', new=[self.ceres_slices[0]]):
      self.assertTrue(self.ceres_node.hasDataForInterval(1740, 2100))

  def test_no_data_exists_if_slice_touches_interval_end(self):
    with patch('ceres.CeresNode.slices', new=[self.ceres_slices[0]]):
      self.assertFalse(self.ceres_node.hasDataForInterval(600, 1200))

  def test_no_data_exists_if_slice_touches_interval_start(self):
    with patch('ceres.CeresNode.slices', new=[self.ceres_slices[0]]):
      self.assertFalse(self.ceres_node.hasDataForInterval(1800, 2100))

  def test_compact_returns_empty_if_passed_empty(self):
    self.assertEqual([], self.ceres_node.compact([]))

  def test_compact_filters_null_values(self):
    self.assertEqual([], self.ceres_node.compact([(60, None)]))

  def test_compact_rounds_timestamps_down_to_step(self):
    self.assertEqual([[(600, 0)]], self.ceres_node.compact([(605, 0)]))

  def test_compact_drops_duplicate_timestamps(self):
    datapoints = [(600, 0), (600, 0)]
    compacted = self.ceres_node.compact(datapoints)
    self.assertEqual([[(600, 0.0)]], compacted)

  @skip("XXX: Ceres should keep the last of duplicate points")
  def test_compact_keeps_last_seen_duplicate_timestamp(self):
    datapoints = [(600, 0), (600, 1)]
    compacted = self.ceres_node.compact(datapoints)
    self.assertEqual([[(600, 1.0)]], compacted)

  def test_compact_groups_contiguous_points(self):
    datapoints = [(600, 0), (660, 0), (840, 0)]
    compacted = self.ceres_node.compact(datapoints)
    self.assertEqual([[(600, 0), (660, 0)], [(840, 0)]], compacted)

  def test_write_noops_if_no_datapoints(self):
    with patch('ceres.CeresNode.slices', new=self.ceres_slices):
      self.ceres_node.write([])
      self.assertFalse(self.ceres_slices[0].write.called)

  def test_write_within_first_slice(self):
    datapoints = [(1200, 0.0), (1260, 1.0), (1320, 2.0)]

    with patch('ceres.CeresNode.slices', new=self.ceres_slices):
      self.ceres_node.write(datapoints)
      self.ceres_slices[0].write.assert_called_once_with(datapoints)

  @patch('ceres.CeresSlice.create')
  def test_write_within_first_slice_doesnt_create(self, slice_create_mock):
    datapoints = [(1200, 0.0), (1260, 1.0), (1320, 2.0)]

    with patch('ceres.CeresNode.slices', new=self.ceres_slices):
      self.ceres_node.write(datapoints)
      self.assertFalse(slice_create_mock.called)

  @patch('ceres.CeresSlice.create', new=Mock())
  def test_write_within_first_slice_with_gaps(self):
    datapoints = [(1200, 0.0), (1320, 2.0)]

    with patch('ceres.CeresNode.slices', new=self.ceres_slices):
      self.ceres_node.write(datapoints)

      # sorted most recent first
      calls = [call.write([datapoints[1]]), call.write([datapoints[0]])]
      self.ceres_slices[0].assert_has_calls(calls)

  @patch('ceres.CeresSlice.create', new=Mock())
  def test_write_within_previous_slice(self):
    datapoints = [(720, 0.0), (780, 2.0)]

    with patch('ceres.CeresNode.slices', new=self.ceres_slices):
      self.ceres_node.write(datapoints)

      # 2nd slice has this range
      self.ceres_slices[1].write.assert_called_once_with(datapoints)

  @patch('ceres.CeresSlice.create')
  def test_write_within_previous_slice_doesnt_create(self, slice_create_mock):
    datapoints = [(720, 0.0), (780, 2.0)]

    with patch('ceres.CeresNode.slices', new=self.ceres_slices):
      self.ceres_node.write(datapoints)
      self.assertFalse(slice_create_mock.called)

  @patch('ceres.CeresSlice.create', new=Mock())
  def test_write_within_previous_slice_with_gaps(self):
    datapoints = [(720, 0.0), (840, 2.0)]

    with patch('ceres.CeresNode.slices', new=self.ceres_slices):
      self.ceres_node.write(datapoints)

      calls = [call.write([datapoints[1]]), call.write([datapoints[0]])]
      self.ceres_slices[1].assert_has_calls(calls)

  @patch('ceres.CeresSlice.create', new=Mock())
  def test_write_across_slice_boundaries(self):
    datapoints = [(1080, 0.0), (1140, 1.0), (1200, 2.0), (1260, 3.0)]

    with patch('ceres.CeresNode.slices', new=self.ceres_slices):
      self.ceres_node.write(datapoints)
      self.ceres_slices[0].write.assert_called_once_with(datapoints[2:4])
      self.ceres_slices[1].write.assert_called_once_with(datapoints[0:2])

  @patch('ceres.CeresSlice.create')
  def test_write_before_earliest_slice_creates_new(self, slice_create_mock):
    datapoints = [(300, 0.0)]
    with patch('ceres.CeresNode.slices', new=self.ceres_slices):
      self.ceres_node.write(datapoints)
      slice_create_mock.assert_called_once_with(self.ceres_node, 300, 60)

  @patch('ceres.CeresSlice.create')
  def test_write_before_earliest_slice_writes_to_new_one(self, slice_create_mock):
    datapoints = [(300, 0.0)]
    with patch('ceres.CeresNode.slices', new=self.ceres_slices):
      self.ceres_node.write(datapoints)
      slice_create_mock.return_value.write.assert_called_once_with(datapoints)

  @patch('ceres.CeresSlice.create')
  def test_write_before_earliest_slice_writes_next_slice_too(self, slice_create_mock):
    # slice 0 starts at 600
    datapoints = [(540, 0.0), (600, 0.0)]
    with patch('ceres.CeresNode.slices', new=self.ceres_slices):
      self.ceres_node.write(datapoints)
      self.ceres_slices[1].write.assert_called_once_with([datapoints[1]])

  @patch('ceres.CeresSlice.create')
  def test_create_during_write_clears_slice_cache(self, slice_create_mock):
    self.ceres_node.setSliceCachingBehavior('all')
    self.ceres_node.sliceCache = self.ceres_slices
    datapoints = [(300, 0.0)]
    with patch('ceres.CeresNode.slices', new=self.ceres_slices):
      self.ceres_node.write(datapoints)
      self.assertEquals(None, self.ceres_node.sliceCache)

  @patch('ceres.CeresSlice.create')
  def test_write_past_max_gap_size_creates(self, slice_create_mock):
    datapoints = [(6000, 0.0)]
    with patch('ceres.CeresNode.slices', new=self.ceres_slices):
      with patch.object(self.ceres_slices[0], 'write', side_effect=SliceGapTooLarge):
        self.ceres_node.write(datapoints)

  @patch('ceres.CeresSlice.create')
  def test_write_different_timestep_creates(self, slice_create_mock):
    datapoints = [(600, 0.0)]
    with patch('ceres.CeresNode.slices', new=self.ceres_slices):
      self.ceres_node.timeStep = 10
      self.ceres_node.write(datapoints)
      slice_create_mock.assert_called_once_with(self.ceres_node, 600, 10)


class CeresNodeReadTest(TestCase):
  def setUp(self):
    with patch('ceres.isdir', new=Mock(return_value=True)):
      with patch('ceres.exists', new=Mock(return_value=True)):
        self.ceres_tree = CeresTree('/graphite/storage/ceres')
        self.ceres_node = CeresNode(
            self.ceres_tree,
            'sample_metric',
            '/graphite/storage/ceres/sample_metric')
        self.ceres_node.timeStep = 60

    slice_configs = [
      (1200, 1800, 60),
      (600, 1200, 60)]

    self.ceres_slices = []
    for start, end, step in slice_configs:
      slice_mock = make_slice_mock(start, end, step)
      self.ceres_slices.append(slice_mock)

    self.ceres_slices_patch = patch('ceres.CeresNode.slices', new=iter(self.ceres_slices))
    self.ceres_slices_patch.start()

  def tearDown(self):
    self.ceres_slices_patch.stop()

  def test_read_loads_metadata_if_timestep_unknown(self):
    with patch('ceres.CeresNode.readMetadata', new=Mock(side_effect=Exception))\
      as read_metadata_mock:
      self.ceres_node.timeStep = None
      try:  # Raise Exception as a cheap exit out of the function once we have the call we want
        self.ceres_node.read(600, 660)
      except Exception:
        pass
      read_metadata_mock.assert_called_once_with()

  def test_read_normalizes_from_time(self):
    self.ceres_node.read(1210, 1260)
    self.ceres_slices[0].read.assert_called_once_with(1200, 1260)

  def test_read_normalizes_until_time(self):
    self.ceres_node.read(1200, 1270)
    self.ceres_slices[0].read.assert_called_once_with(1200, 1260)

  def test_read_returns_empty_time_series_if_before_slices(self):
    result = self.ceres_node.read(0, 300)
    self.assertEqual([None] * 5, result.values)

  def test_read_returns_empty_time_series_if_slice_has_no_data(self):
    self.ceres_slices[0].read.side_effect = NoData
    result = self.ceres_node.read(1200, 1500)
    self.assertEqual([None] * 5, result.values)

  def test_read_pads_points_missing_before_series(self):
    result = self.ceres_node.read(540, 1200)
    self.assertEqual([None] + [0] * 10, result.values)

  def test_read_pads_points_missing_after_series(self):
    result = self.ceres_node.read(1200, 1860)
    self.assertEqual(None, result.values[-1])

  def test_read_goes_across_slices(self):
    self.ceres_node.read(900, 1500)
    self.ceres_slices[0].read.assert_called_once_with(1200, 1500)
    self.ceres_slices[1].read.assert_called_once_with(900, 1200)

  def test_read_across_slices_merges_results(self):
    result = self.ceres_node.read(900, 1500)
    self.assertEqual([0] * 10, result.values)

  def test_read_pads_points_missing_after_series_across_slices(self):
    result = self.ceres_node.read(900, 1860)
    self.assertEqual(None, result.values[-1])

  def test_read_pads_points_missing_between_slices(self):
    self.ceres_slices[1] = make_slice_mock(600, 1140, 60)
    result = self.ceres_node.read(900, 1500)
    self.assertEqual([0] * 4 + [None] + [0] * 5, result.values)


class CeresSliceTest(TestCase):
  def setUp(self):
    with patch('ceres.isdir', new=Mock(return_value=True)):
      with patch('ceres.exists', new=Mock(return_value=True)):
        self.ceres_tree = CeresTree('/graphite/storage/ceres')
        self.ceres_node = CeresNode(
            self.ceres_tree,
            'sample_metric',
            '/graphite/storage/ceres/sample_metric')

  def test_init_sets_fspath_name(self):
    ceres_slice = CeresSlice(self.ceres_node, 0, 60)
    self.assertTrue(ceres_slice.fsPath.endswith('0@60.slice'))

  @patch('ceres.getsize')
  def test_end_time_calculated_via_filesize(self, getsize_mock):
    getsize_mock.return_value = DATAPOINT_SIZE * 300
    ceres_slice = CeresSlice(self.ceres_node, 0, 60)
    # 300 points at 60 sec per point
    self.assertEqual(300 * 60, ceres_slice.endTime)

  @patch('ceres.exists')
  def test_delete_before_raises_if_deleted(self, exists_mock):
    exists_mock.return_value = False
    ceres_slice = CeresSlice(self.ceres_node, 0, 60)
    self.assertRaises(SliceDeleted, ceres_slice.deleteBefore, 60)

  @patch('ceres.exists', Mock(return_value=True))
  @patch.object(builtins, 'open', new_callable=mock_open)
  def test_delete_before_returns_if_time_earlier_than_start(self, open_mock):
    ceres_slice = CeresSlice(self.ceres_node, 300, 60)
    # File starts at timestamp 300, delete points before timestamp 60
    ceres_slice.deleteBefore(60)
    open_mock.assert_has_calls([])  # no calls

  @patch('ceres.exists', Mock(return_value=True))
  @patch.object(builtins, 'open', new_callable=mock_open)
  def test_delete_before_returns_if_time_less_than_step_earlier_than_start(self, open_mock):
    ceres_slice = CeresSlice(self.ceres_node, 300, 60)
    ceres_slice.deleteBefore(299)
    open_mock.assert_has_calls([])

  @patch('ceres.exists', Mock(return_value=True))
  @patch.object(builtins, 'open', new_callable=mock_open)
  def test_delete_before_returns_if_time_same_as_start(self, open_mock):
    ceres_slice = CeresSlice(self.ceres_node, 300, 60)
    ceres_slice.deleteBefore(300)
    open_mock.assert_has_calls([])

  @patch('ceres.exists', Mock(return_value=True))
  @patch('ceres.os.rename', Mock(return_value=True))
  def test_delete_before_clears_slice_cache(self):
    ceres_slice = CeresSlice(self.ceres_node, 300, 60)
    open_mock = mock_open(read_data='foo')  # needs to be non-null for this test
    with patch.object(builtins, 'open', open_mock):
      with patch('ceres.CeresNode.clearSliceCache') as clear_slice_cache_mock:
        ceres_slice.deleteBefore(360)
        clear_slice_cache_mock.assert_called_once_with()

  @patch('ceres.exists', Mock(return_value=True))
  @patch.object(builtins, 'open', new_callable=mock_open)
  def test_delete_before_deletes_file_if_no_more_data(self, open_mock):
    ceres_slice = CeresSlice(self.ceres_node, 300, 60)
    with patch('ceres.os.unlink') as unlink_mock:
      try:
        ceres_slice.deleteBefore(360)
      except Exception:
        pass
      self.assertTrue(unlink_mock.called)

  @patch('ceres.exists', Mock(return_value=True))
  @patch('ceres.os.unlink', Mock())
  @patch.object(builtins, 'open', new_callable=mock_open)
  def test_delete_before_raises_slice_deleted_if_no_more_data(self, open_mock):
    ceres_slice = CeresSlice(self.ceres_node, 300, 60)
    self.assertRaises(SliceDeleted, ceres_slice.deleteBefore, 360)

  @patch('ceres.exists', Mock(return_value=True))
  @patch('ceres.os.rename', Mock())
  def test_delete_before_seeks_to_time(self):
    ceres_slice = CeresSlice(self.ceres_node, 300, 60)
    open_mock = mock_open(read_data='foo')
    with patch.object(builtins, 'open', open_mock) as open_mock:
      ceres_slice.deleteBefore(360)
      # Seek from 300 (start of file) to 360 (1 datapointpoint)
      open_mock.return_value.seek.assert_any_call(1 * DATAPOINT_SIZE)

  @patch('ceres.exists', Mock(return_value=True))
  def test_slices_are_sortable(self):
    ceres_slices = [
      CeresSlice(self.ceres_node, 300, 60),
      CeresSlice(self.ceres_node, 600, 60),
      CeresSlice(self.ceres_node, 0, 60)]

    expected_order = [0, 300, 600]
    result_order = [slice.startTime for slice in sorted(ceres_slices)]
    self.assertEqual(expected_order, result_order)


class CeresSliceWriteTest(TestCase):
  def setUp(self):
    with patch('ceres.isdir', new=Mock(return_value=True)):
      with patch('ceres.exists', new=Mock(return_value=True)):
        self.ceres_tree = CeresTree('/graphite/storage/ceres')
        self.ceres_node = CeresNode(
            self.ceres_tree,
            'sample_metric',
            '/graphite/storage/ceres/sample_metric')
    self.ceres_slice = CeresSlice(self.ceres_node, 300, 60)

  @patch('ceres.getsize', Mock(side_effect=OSError))
  def test_raises_os_error_if_not_enoent(self):
    self.assertRaises(OSError, self.ceres_slice.write, [(0, 0)])

  @patch('ceres.getsize', Mock(side_effect=OSError(errno.ENOENT, 'foo')))
  def test_raises_slice_deleted_oserror_enoent(self):
    self.assertRaises(SliceDeleted, self.ceres_slice.write, [(0, 0)])

  @patch('ceres.getsize', Mock(return_value=0))
  @patch.object(builtins, 'open', mock_open())
  def test_raises_slice_gap_too_large_when_it_is(self):
    # one point over the max
    new_time = self.ceres_slice.startTime + self.ceres_slice.timeStep * (MAX_SLICE_GAP + 1)
    datapoint = (new_time, 0)
    self.assertRaises(SliceGapTooLarge, self.ceres_slice.write, [datapoint])

  @patch('ceres.getsize', Mock(return_value=0))
  @patch.object(builtins, 'open', mock_open())
  def test_doesnt_raise_slice_gap_too_large_when_it_isnt(self):
    new_time = self.ceres_slice.startTime + self.ceres_slice.timeStep * (MAX_SLICE_GAP - 1)
    datapoint = (new_time, 0)
    try:
      self.ceres_slice.write([datapoint])
    except SliceGapTooLarge:
      self.fail("SliceGapTooLarge raised")

  @patch('ceres.getsize', Mock(return_value=DATAPOINT_SIZE * 100))
  @patch.object(builtins, 'open', mock_open())
  def test_doesnt_raise_slice_gap_when_newer_points_exist(self):
    new_time = self.ceres_slice.startTime + self.ceres_slice.timeStep * (MAX_SLICE_GAP + 1)
    datapoint = (new_time, 0)
    try:
      self.ceres_slice.write([datapoint])
    except SliceGapTooLarge:
      self.fail("SliceGapTooLarge raised")

  @patch('ceres.getsize', Mock(return_value=0))
  @patch.object(builtins, 'open', new_callable=mock_open)
  def test_raises_ioerror_if_seek_hits_ioerror(self, open_mock):
    open_mock.return_value.seek.side_effect = IOError
    self.assertRaises(IOError, self.ceres_slice.write, [(300, 0)])

  @patch('ceres.getsize', Mock(return_value=0))
  @patch.object(builtins, 'open', new_callable=mock_open)
  def test_opens_file_as_binary(self, open_mock):
    self.ceres_slice.write([(300, 0)])
    # call_args = (args, kwargs)
    self.assertTrue(open_mock.call_args[0][1].endswith('b'))

  @patch('ceres.getsize', Mock(return_value=0))
  @patch.object(builtins, 'open', new_callable=mock_open)
  def test_seeks_to_the_correct_offset_first_point(self, open_mock):
    self.ceres_slice.write([(300, 0)])
    open_mock.return_value.seek.assert_called_once_with(0)

  @patch('ceres.getsize', Mock(return_value=1 * DATAPOINT_SIZE))
  @patch.object(builtins, 'open', new_callable=mock_open)
  def test_seeks_to_the_correct_offset_next_point(self, open_mock):
    self.ceres_slice.write([(360, 0)])
    # 2nd point in the file
    open_mock.return_value.seek.assert_called_once_with(DATAPOINT_SIZE)

  @patch('ceres.getsize', Mock(return_value=1 * DATAPOINT_SIZE))
  @patch.object(builtins, 'open', new_callable=mock_open)
  def test_seeks_to_the_next_empty_offset_one_point_gap(self, open_mock):
    # Gaps are written out as NaNs so the offset we expect is the beginning
    # of the gap, not the offset of the point itself
    self.ceres_slice.write([(420, 0)])
    open_mock.return_value.seek.assert_called_once_with(1 * DATAPOINT_SIZE)

  @patch('ceres.getsize', Mock(return_value=0))
  @patch.object(builtins, 'open', new_callable=mock_open)
  def test_correct_size_written_first_point(self, open_mock):
    self.ceres_slice.write([(300, 0)])
    self.assertEqual(1 * DATAPOINT_SIZE, len(fetch_mock_open_writes(open_mock)))

  @patch('ceres.getsize', Mock(return_value=1 * DATAPOINT_SIZE))
  @patch.object(builtins, 'open', new_callable=mock_open)
  def test_correct_size_written_next_point(self, open_mock):
    self.ceres_slice.write([(360, 0)])
    self.assertEqual(1 * DATAPOINT_SIZE, len(fetch_mock_open_writes(open_mock)))

  @patch('ceres.getsize', Mock(return_value=1 * DATAPOINT_SIZE))
  @patch.object(builtins, 'open', new_callable=mock_open)
  def test_correct_size_written_one_point_gap(self, open_mock):
    self.ceres_slice.write([(420, 0)])
    # one empty point, one real point = two points total written
    self.assertEqual(2 * DATAPOINT_SIZE, len(fetch_mock_open_writes(open_mock)))

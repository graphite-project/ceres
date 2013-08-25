from unittest import TestCase
from mock import ANY, Mock, call, mock_open, patch

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
        self.ceres_node.timeStep = 60

    slice_configs = [
      ( 1200, 1800, 60 ),
      ( 600, 1200, 60 )]

    self.ceres_slices = []
    for start, end, step in slice_configs:
      slice_mock = Mock(spec=CeresSlice)
      slice_mock.startTime = start
      slice_mock.endTime = end
      slice_mock.timeStep = step

      self.ceres_slices.append(slice_mock)


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

  @patch('ceres.exists', new=Mock(return_Value=True))
  def test_read_slices_parses_slice_filenames(self):
    listdir_mock = Mock(return_value=['0@60.slice', '0@300.slice'])
    with patch('ceres.os.listdir', new=listdir_mock):
      slice_infos = self.ceres_node.readSlices()
      self.assertTrue((0,60) in slice_infos)
      self.assertTrue((0,300) in slice_infos)

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
      slice_timestamps = [ s[0] for s in slice_infos ]
      self.assertEqual([600,320,120,0,0], slice_timestamps)

  def test_no_data_exists_if_no_slices_exist(self):
    with patch('ceres.CeresNode.readSlices', new=Mock(return_value=[])):
      self.assertFalse(self.ceres_node.hasDataForInterval(0,60))

  def test_no_data_exists_if_no_slices_exist_and_no_time_specified(self):
    with patch('ceres.CeresNode.readSlices', new=Mock(return_value=[])):
      self.assertFalse(self.ceres_node.hasDataForInterval(None,None))

  def test_data_exists_if_slices_exist_and_no_time_specified(self):
    with patch('ceres.CeresNode.slices', new=self.ceres_slices):
      self.assertTrue(self.ceres_node.hasDataForInterval(None,None))

  def test_data_exists_if_slice_covers_interval_completely(self):
    with patch('ceres.CeresNode.slices', new=[self.ceres_slices[0]]):
      self.assertTrue(self.ceres_node.hasDataForInterval(1200,1800))

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
    self.assertEqual([], self.ceres_node.compact([(60,None)]))

  def test_compact_rounds_timestamps_down_to_step(self):
    self.assertEqual([[(600,0)]], self.ceres_node.compact([(605,0)]))

  def test_compact_drops_duplicate_timestamps(self):
    datapoints = [ (600, 0), (600, 0) ]
    compacted = self.ceres_node.compact(datapoints)
    self.assertEqual([[(600, 0)]], compacted)

  def test_compact_groups_contiguous_points(self):
    datapoints = [ (600, 0), (660, 0), (840,0) ]
    compacted = self.ceres_node.compact(datapoints)
    self.assertEqual([[(600, 0), (660,0)], [(840,0)]], compacted)

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
    datapoints = [ (1200,0.0), (1320,2.0) ]

    with patch('ceres.CeresNode.slices', new=self.ceres_slices):
      self.ceres_node.write(datapoints)

      # sorted most recent first
      calls = [call.write([datapoints[1]]), call.write([datapoints[0]])]
      self.ceres_slices[0].assert_has_calls(calls)

  @patch('ceres.CeresSlice.create', new=Mock())
  def test_write_within_previous_slice(self):
    datapoints = [ (720,0.0), (780,2.0) ]

    with patch('ceres.CeresNode.slices', new=self.ceres_slices):
      self.ceres_node.write(datapoints)

      # 2nd slice has this range
      self.ceres_slices[1].write.assert_called_once_with(datapoints)

  @patch('ceres.CeresSlice.create')
  def test_write_within_previous_slice_doesnt_create(self, slice_create_mock):
    datapoints = [ (720,0.0), (780,2.0) ]

    with patch('ceres.CeresNode.slices', new=self.ceres_slices):
      self.ceres_node.write(datapoints)
      self.assertFalse(slice_create_mock.called)

  @patch('ceres.CeresSlice.create', new=Mock())
  def test_write_within_previous_slice_with_gaps(self):
    datapoints = [ (720,0.0), (840,2.0) ]

    with patch('ceres.CeresNode.slices', new=self.ceres_slices):
      self.ceres_node.write(datapoints)

      calls = [call.write([datapoints[1]]), call.write([datapoints[0]])]
      self.ceres_slices[1].assert_has_calls(calls)

  @patch('ceres.CeresSlice.create', new=Mock())
  def test_write_across_slice_boundaries(self):
    datapoints = [ (1080,0.0), (1140,1.0), (1200, 2.0), (1260, 3.0) ]

    with patch('ceres.CeresNode.slices', new=self.ceres_slices):
      self.ceres_node.write(datapoints)
      self.ceres_slices[0].write.assert_called_once_with(datapoints[2:4])
      self.ceres_slices[1].write.assert_called_once_with(datapoints[0:2])

  @patch('ceres.CeresSlice.create')
  def test_write_before_earliest_slice_creates_new(self, slice_create_mock):
    datapoints = [ (300, 0.0) ]
    with patch('ceres.CeresNode.slices', new=self.ceres_slices):
      self.ceres_node.write(datapoints)
      slice_create_mock.assert_called_once_with(self.ceres_node, 300, 60)

  @patch('ceres.CeresSlice.create')
  def test_write_before_earliest_slice_writes_to_new_one(self, slice_create_mock):
    datapoints = [ (300, 0.0) ]
    with patch('ceres.CeresNode.slices', new=self.ceres_slices):
      self.ceres_node.write(datapoints)
      slice_create_mock.return_value.write.assert_called_once_with(datapoints)

  @patch('ceres.CeresSlice.create')
  def test_create_during_write_clears_slice_cache(self, slice_create_mock):
    self.ceres_node.setSliceCachingBehavior('all')
    self.ceres_node.sliceCache = self.ceres_slices
    datapoints = [ (300, 0.0) ]
    with patch('ceres.CeresNode.slices', new=self.ceres_slices):
      self.ceres_node.write(datapoints)
      self.assertEquals(None, self.ceres_node.sliceCache)


class CeresSliceTest(TestCase):
  def setUp(self):
    with patch('ceres.isdir', new=Mock(return_value=True)):
      with patch('ceres.exists', new=Mock(return_value=True)):
        self.ceres_tree = CeresTree('/graphite/storage/ceres')
        self.ceres_node = CeresNode(self.ceres_tree, 'sample_metric', '/graphite/storage/ceres/sample_metric')

  def test_init_sets_fspath_name(self):
    ceres_slice = CeresSlice(self.ceres_node, 0, 60)
    self.assertTrue(ceres_slice.fsPath.endswith('0@60.slice'))



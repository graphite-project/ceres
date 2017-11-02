# Copyright 2011 Chris Davis
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#

# Ceres requires Python 2.7 or newer
import itertools
import os
import struct
import json
import errno
from math import isnan
from os.path import isdir, exists, join, dirname, abspath, getsize, getmtime
from glob import glob
from bisect import bisect_left

izip = getattr(itertools, 'izip', zip)

try:
  import fcntl
  CAN_LOCK = True
except ImportError:
  CAN_LOCK = False

LOCK_WRITES = False
TIMESTAMP_FORMAT = "!L"
TIMESTAMP_SIZE = struct.calcsize(TIMESTAMP_FORMAT)
DATAPOINT_FORMAT = "!d"
DATAPOINT_SIZE = struct.calcsize(DATAPOINT_FORMAT)
NAN = float('nan')
PACKED_NAN = struct.pack(DATAPOINT_FORMAT, NAN)
MAX_SLICE_GAP = 80
DEFAULT_TIMESTEP = 60
DEFAULT_NODE_CACHING_BEHAVIOR = 'all'
DEFAULT_SLICE_CACHING_BEHAVIOR = 'none'
SLICE_AGGREGATION_METHODS = ['average', 'sum', 'last', 'max', 'min']
SLICE_PERMS = 0o644
DIR_PERMS = 0o755


class CeresTree(object):
  """Represents a tree of Ceres metrics contained within a single path on disk
  This is the primary Ceres API.

  :param root: The directory root of the Ceres tree

  .. note:: Use :func:`createTree` to initialize and instantiate a new CeresTree

  .. seealso:: :func:`setDefaultNodeCachingBehavior` to adjust caching behavior
  """
  def __init__(self, root):
    if isdir(root):
      self.root = abspath(root)
    else:
      raise ValueError("Invalid root directory '%s'" % root)
    self.nodeCache = {}
    self.nodeCachingBehavior = DEFAULT_NODE_CACHING_BEHAVIOR

  def __repr__(self):
    return "<CeresTree[0x%x]: %s>" % (id(self), self.root)
  __str__ = __repr__

  @classmethod
  def createTree(cls, root, **props):
    """Create and returns a new Ceres tree with the given properties

    :param root: The root directory of the new Ceres tree
    :param \*\*props: Arbitrary key-value properties to store as tree metadata

    :returns: :class:`CeresTree`
    """

    ceresDir = join(root, '.ceres-tree')
    if not isdir(ceresDir):
      os.makedirs(ceresDir, DIR_PERMS)

    for prop, value in props.items():
      propFile = join(ceresDir, prop)
      with open(propFile, 'w') as fh:
        fh.write(str(value))

    return cls(root)

  def walk(self, **kwargs):
    """Iterate through the nodes contained in this :class:`CeresTree`

      :param \*\*kwargs: Options to pass to :func:`os.walk`

      :returns: An iterator yielding :class:`CeresNode` objects
    """
    for (fsPath, subdirs, filenames) in os.walk(self.root, **kwargs):
      if CeresNode.isNodeDir(fsPath):
        nodePath = self.getNodePath(fsPath)
        yield CeresNode(self, nodePath, fsPath)

  def getFilesystemPath(self, nodePath):
    """Get the on-disk path of a Ceres node given a metric name

    :param nodePath: A metric name e.g. ``carbon.agents.graphite-a.cpuUsage``

    :returns: The Ceres node path on disk"""
    return join(self.root, nodePath.replace('.', os.sep))

  def getNodePath(self, fsPath):
    """Get the metric name of a Ceres node given the on-disk path

    :param fsPath: The filesystem path of a Ceres node

    :returns: A metric name

    :raises ValueError: When `fsPath` is not a path within the :class:`CeresTree`
    """
    fsPath = abspath(fsPath)
    if not fsPath.startswith(self.root):
      raise ValueError("path '%s' not beneath tree root '%s'" % (fsPath, self.root))

    nodePath = fsPath[len(self.root):].strip(os.sep).replace(os.sep, '.')
    return nodePath

  def hasNode(self, nodePath):
    """Returns whether the Ceres tree contains the given metric

    :param nodePath: A metric name e.g. ``carbon.agents.graphite-a.cpuUsage``

    :returns: `True` or `False`"""
    return isdir(self.getFilesystemPath(nodePath))

  def setNodeCachingBehavior(self, behavior):
    """Set node caching behavior.

    :param behavior: See :func:`getNode` for valid behavior values
    """
    behavior = behavior.lower()
    if behavior not in ('none', 'all'):
      raise ValueError("invalid caching behavior '%s'" % behavior)

    self.nodeCachingBehavior = behavior
    self.nodeCache = {}

  def getNode(self, nodePath):
    """Returns a Ceres node given a metric name. Because nodes are looked up in
    every read and write, a caching mechanism is provided. Cache behavior is set
    using :func:`setNodeCachingBehavior` and defaults to the value set in
    ``DEFAULT_NODE_CACHING_BEHAVIOR``

    The following behaviors are available:

    * `none` - Node is read from the filesystem at every access.
    * `all` (default) - All nodes are cached.

      :param nodePath: A metric name

      :returns: :class:`CeresNode` or `None`
    """
    if self.nodeCachingBehavior == 'all':
      if nodePath not in self.nodeCache:
        fsPath = self.getFilesystemPath(nodePath)
        if CeresNode.isNodeDir(fsPath):
          self.nodeCache[nodePath] = CeresNode(self, nodePath, fsPath)
        else:
          return None

      return self.nodeCache[nodePath]

    elif self.nodeCachingBehavior == 'none':
      fsPath = self.getFilesystemPath(nodePath)
      if CeresNode.isNodeDir(fsPath):
        return CeresNode(self, nodePath, fsPath)
      else:
        return None

    else:
      raise ValueError("invalid caching behavior configured '%s'" % self.nodeCachingBehavior)

  def find(self, nodePattern, fromTime=None, untilTime=None):
    """Find nodes which match a wildcard pattern, optionally filtering on
    a time range

      :param nodePattern: A glob-style metric wildcard
      :param fromTime: Optional interval start time in unix-epoch.
      :param untilTime: Optional interval end time in unix-epoch.

      :returns: An iterator yielding :class:`CeresNode` objects
    """
    for fsPath in glob(self.getFilesystemPath(nodePattern)):
      if CeresNode.isNodeDir(fsPath):
        nodePath = self.getNodePath(fsPath)
        node = self.getNode(nodePath)

        if fromTime is None and untilTime is None:
          yield node
        elif node.hasDataForInterval(fromTime, untilTime):
          yield node

  def createNode(self, nodePath, **properties):
    """Creates a new metric given a new metric name and optional per-node metadata
      :param nodePath: The new metric name.
      :param \*\*properties: Arbitrary key-value properties to store as metric metadata.

      :returns: :class:`CeresNode`
    """
    return CeresNode.create(self, nodePath, **properties)

  def store(self, nodePath, datapoints):
    """Store a list of datapoints associated with a metric
      :param nodePath: The metric name to write to e.g. ``carbon.agents.graphite-a.cpuUsage``
      :param datapoints: A list of datapoint tuples: ``[(timestamp, value), ...]``
    """
    node = self.getNode(nodePath)

    if node is None:
      raise NodeNotFound("The node '%s' does not exist in this tree" % nodePath)

    node.write(datapoints)

  def fetch(self, nodePath, fromTime, untilTime):
    """Fetch data within a given interval from the given metric

      :param nodePath: The metric name to fetch from
      :param fromTime: Requested interval start time in unix-epoch.
      :param untilTime: Requested interval end time in unix-epoch.

      :returns: :class:`TimeSeriesData`
      :raises: :class:`NodeNotFound`, :class:`InvalidRequest`
    """
    node = self.getNode(nodePath)

    if not node:
      raise NodeNotFound("the node '%s' does not exist in this tree" % nodePath)

    return node.read(fromTime, untilTime)


class CeresNode(object):
  """A :class:`CeresNode` represents a single time-series metric of a given `timeStep`
(its seconds-per-point resolution) and containing arbitrary key-value metadata.

A :class:`CeresNode` is associated with its most precise `timeStep`. This `timeStep` is the finest
resolution that can be used for writing, though a :class:`CeresNode` can contain and read data with
other, less-precise `timeStep` values in its underlying :class:`CeresSlice` data.

  :param tree: The :class:`CeresTree` this node is associated with
  :param nodePath: The name of the metric this node represents
  :param fsPath: The filesystem path of this metric

  .. note:: This class generally should be instantiated through use of :class:`CeresTree`. See
            :func:`CeresTree.createNode` and :func:`CeresTree.getNode`

  .. seealso:: :func:`setDefaultSliceCachingBehavior` to adjust caching behavior
  """
  __slots__ = ('tree', 'nodePath', 'fsPath',
               'metadataFile', 'timeStep', 'aggregationMethod',
               'sliceCache', 'sliceCachingBehavior')

  def __init__(self, tree, nodePath, fsPath):
    self.tree = tree
    self.nodePath = nodePath
    self.fsPath = fsPath
    self.metadataFile = join(fsPath, '.ceres-node')
    self.timeStep = None
    self.aggregationMethod = 'average'
    self.sliceCache = None
    self.sliceCachingBehavior = DEFAULT_SLICE_CACHING_BEHAVIOR

  def __repr__(self):
    return "<CeresNode[0x%x]: %s>" % (id(self), self.nodePath)
  __str__ = __repr__

  @classmethod
  def create(cls, tree, nodePath, **properties):
    """Create a new :class:`CeresNode` on disk with the specified properties.

    :param tree: The :class:`CeresTree` this node is associated with
    :param nodePath: The name of the metric this node represents
    :param \*\*properties: A set of key-value properties to be associated with this node

A :class:`CeresNode` always has the `timeStep` property which is an integer value representing
the precision of the node in seconds-per-datapoint. E.g. a value of ``60`` represents one datapoint
per minute. If no `timeStep` is specified at creation, the value of ``ceres.DEFAULT_TIMESTEP`` is
used

    :returns: :class:`CeresNode`
    """
    # Create the node directory
    fsPath = tree.getFilesystemPath(nodePath)
    os.makedirs(fsPath, DIR_PERMS)

    properties['timeStep'] = properties.get('timeStep', DEFAULT_TIMESTEP)
    # Create the initial metadata
    node = cls(tree, nodePath, fsPath)
    node.writeMetadata(properties)

    # Create the initial data file
    # timeStep = properties['timeStep']
    # now = int( time.time() )
    # baseTime = now - (now % timeStep)
    # slice = CeresSlice.create(node, baseTime, timeStep)

    return node

  @staticmethod
  def isNodeDir(path):
    """Tests whether the given path is a :class:`CeresNode`

    :param path: Path to test
    :returns `True` or `False`
    """
    return isdir(path) and exists(join(path, '.ceres-node'))

  @classmethod
  def fromFilesystemPath(cls, fsPath):
    """Instantiate a :class:`CeresNode` from the on-disk path of an existing node

    :params fsPath: The filesystem path of an existing node
    :returns: :class:`CeresNode`
    """
    dirPath = dirname(fsPath)

    while True:
      ceresDir = join(dirPath, '.ceres-tree')
      if isdir(ceresDir):
        tree = CeresTree(dirPath)
        nodePath = tree.getNodePath(fsPath)
        return cls(tree, nodePath, fsPath)

      dirPath = dirname(dirPath)

      if dirPath == '/':
        raise ValueError("the path '%s' is not in a ceres tree" % fsPath)

  @property
  def slice_info(self):
    """A property providing a list of current information about each slice

    :returns: ``[(startTime, endTime, timeStep), ...]``
    """
    return [(slice.startTime, slice.endTime, slice.timeStep) for slice in self.slices]

  def readMetadata(self):
    """Update node metadata from disk

    :raises: :class:`CorruptNode`
    """
    with open(self.metadataFile, 'r') as fh:
      try:
        metadata = json.load(fh)
        self.timeStep = int(metadata['timeStep'])
        if metadata.get('aggregationMethod'):
          self.aggregationMethod = metadata['aggregationMethod']
        return metadata
      except (KeyError, IOError, ValueError) as e:
        raise CorruptNode(self, "Unable to parse node metadata: %s" % e.args)

  def writeMetadata(self, metadata):
    """Writes new metadata to disk

    :param metadata: a JSON-serializable dict of node metadata
    """
    self.timeStep = int(metadata['timeStep'])
    with open(self.metadataFile, 'w') as fh:
      json.dump(metadata, fh)

  @property
  def slices(self):
    """A property providing access to information about this node's underlying slices. Because this
information is accessed in every read and write, a caching mechanism is provided. Cache behavior is
set using :func:`setSliceCachingBehavior` and defaults to the value set in
``DEFAULT_SLICE_CACHING_BEHAVIOR``

The following behaviors are available:

* `none` (default) - Slice information is read from the filesystem at every access
* `latest` - The latest slice is served from cache, all others from disk. Reads and writes of recent
  data are most likely to be in the latest slice
* `all` - All slices are cached. The cache is only refreshed on new slice creation or deletion

    :returns: ``[(startTime, timeStep), ...]``
    """
    if self.sliceCache:
      if self.sliceCachingBehavior == 'all':
        for slice in self.sliceCache:
          yield slice

      elif self.sliceCachingBehavior == 'latest':
        yield self.sliceCache
        infos = self.readSlices()
        for info in infos[1:]:
          yield CeresSlice(self, *info)

    else:
      if self.sliceCachingBehavior == 'all':
        self.sliceCache = [CeresSlice(self, *info) for info in self.readSlices()]
        for slice in self.sliceCache:
          yield slice

      elif self.sliceCachingBehavior == 'latest':
        infos = self.readSlices()
        if infos:
          self.sliceCache = CeresSlice(self, *infos[0])
          yield self.sliceCache

        for info in infos[1:]:
          yield CeresSlice(self, *info)

      elif self.sliceCachingBehavior == 'none':
        for info in self.readSlices():
          yield CeresSlice(self, *info)

      else:
        raise ValueError("invalid caching behavior configured '%s'" % self.sliceCachingBehavior)

  def readSlices(self):
    """Read slice information from disk

    :returns: ``[(startTime, timeStep), ...]``
    """
    if not exists(self.fsPath):
      raise NodeDeleted()

    slice_info = []
    for filename in os.listdir(self.fsPath):
      if filename.endswith('.slice'):
        startTime, timeStep = filename[:-6].split('@')
        slice_info.append((int(startTime), int(timeStep)))

    slice_info.sort(reverse=True)
    return slice_info

  def setSliceCachingBehavior(self, behavior):
    """Set slice caching behavior.

    :param behavior: See :func:`slices` for valid behavior values
    """
    behavior = behavior.lower()
    if behavior not in ('none', 'all', 'latest'):
      raise ValueError("invalid caching behavior '%s'" % behavior)

    self.sliceCachingBehavior = behavior
    self.sliceCache = None

  def clearSliceCache(self):
    """Clear slice cache, forcing a refresh from disk at the next access"""
    self.sliceCache = None

  def hasDataForInterval(self, fromTime, untilTime):
    """Test whether this node has any data in the given time interval. All slices are inspected
which will trigger a read of slice information from disk if slice cache behavior is set to `latest`
or `none` (See :func:`slices`)

    :param fromTime: Beginning of interval in unix epoch seconds
    :param untilTime: End of interval in unix epoch seconds
    :returns `True` or `False`
    """
    slices = list(self.slices)
    if not slices:
      return False

    earliestData = slices[-1].startTime
    latestData = slices[0].endTime

    return ((fromTime is None) or (fromTime < latestData)) and \
           ((untilTime is None) or (untilTime > earliestData))

  def read(self, fromTime, untilTime):
    """Read data from underlying slices and return as a single time-series

    :param fromTime: Beginning of interval in unix epoch seconds
    :param untilTime: End of interval in unix epoch seconds
    :returns: :class:`TimeSeriesData`
    """
    if self.timeStep is None:
      self.readMetadata()

    # Normalize the timestamps to fit proper intervals
    fromTime = int(fromTime - (fromTime % self.timeStep))
    untilTime = int(untilTime - (untilTime % self.timeStep))

    sliceBoundary = None  # to know when to split up queries across slices
    resultValues = []
    earliestData = None
    timeStep = self.timeStep
    method = self.aggregationMethod

    for slice in self.slices:
      # If there was a prior slice covering the requested interval, dont ask for that data again
      if (sliceBoundary is not None) and untilTime > sliceBoundary:
        requestUntilTime = sliceBoundary
      else:
        requestUntilTime = untilTime

      # if the requested interval starts after the start of this slice
      if fromTime >= slice.startTime:
        try:
          series = slice.read(fromTime, requestUntilTime)
        except NoData:
          break

        if series.timeStep != timeStep:
          if len(resultValues) == 0:
            # First slice holding series data, this becomes the default timeStep.
            timeStep = series.timeStep
          elif series.timeStep < timeStep:
            # Series is at a different precision, aggregate to fit our current set.
            series.values = aggregateSeries(method, series.timeStep, timeStep, series.values)
          else:
            # Normalize current set to fit new series data.
            resultValues = aggregateSeries(method, timeStep, series.timeStep, resultValues)
            timeStep = series.timeStep

        earliestData = series.startTime

        rightMissing = (requestUntilTime - series.endTime) // timeStep
        rightNulls = [None for i in range(rightMissing)]
        resultValues = series.values + rightNulls + resultValues
        break

      # or if slice contains data for part of the requested interval
      elif untilTime >= slice.startTime:
        try:
          series = slice.read(slice.startTime, requestUntilTime)
        except NoData:
          continue

        if series.timeStep != timeStep:
          if len(resultValues) == 0:
            # First slice holding series data, this becomes the default timeStep.
            timeStep = series.timeStep
          elif series.timeStep < timeStep:
            # Series is at a different precision, aggregate to fit our current set.
            series.values = aggregateSeries(method, series.timeStep, timeStep, series.values)
          else:
            # Normalize current set to fit new series data.
            resultValues = aggregateSeries(method, timeStep, series.timeStep, resultValues)
            timeStep = series.timeStep

        earliestData = series.startTime

        rightMissing = (requestUntilTime - series.endTime) // timeStep
        rightNulls = [None for i in range(rightMissing)]
        resultValues = series.values + rightNulls + resultValues

      # this is the right-side boundary on the next iteration
      sliceBoundary = slice.startTime

    # The end of the requested interval predates all slices
    if earliestData is None:
      missing = int(untilTime - fromTime) // timeStep
      resultValues = [None for i in range(missing)]

    # Left pad nulls if the start of the requested interval predates all slices
    else:
      leftMissing = (earliestData - fromTime) // timeStep
      leftNulls = [None for i in range(leftMissing)]
      resultValues = leftNulls + resultValues

    return TimeSeriesData(fromTime, untilTime, timeStep, resultValues)

  def write(self, datapoints):
    """Writes datapoints to underlying slices. Datapoints that round to the same timestamp for the
node's `timeStep` will be treated as duplicates and dropped.

      :param datapoints: List of datapoint tuples ``[(timestamp, value), ...]``
    """
    if self.timeStep is None:
      self.readMetadata()

    if not datapoints:
      return

    sequences = self.compact(datapoints)
    needsEarlierSlice = []  # keep track of sequences that precede all existing slices

    while sequences:
      sequence = sequences.pop()
      timestamps = [t for t, v in sequence]
      beginningTime = timestamps[0]
      endingTime = timestamps[-1]
      sliceBoundary = None  # used to prevent writing sequences across slice boundaries
      slicesExist = False

      for slice in self.slices:
        if slice.timeStep != self.timeStep:
          continue

        slicesExist = True

        # truncate sequence so it doesn't cross the slice boundaries
        if beginningTime >= slice.startTime:
          if sliceBoundary is None:
            sequenceWithinSlice = sequence
          else:
            # index of highest timestamp that doesn't exceed sliceBoundary
            boundaryIndex = bisect_left(timestamps, sliceBoundary)
            sequenceWithinSlice = sequence[:boundaryIndex]

          try:
            slice.write(sequenceWithinSlice)
          except SliceGapTooLarge:
            newSlice = CeresSlice.create(self, beginningTime, slice.timeStep)
            newSlice.write(sequenceWithinSlice)
            self.sliceCache = None
          except SliceDeleted:
            self.sliceCache = None
            self.write(datapoints)  # recurse to retry
            return

          sequence = []
          break

        # sequence straddles the current slice, write the right side
        # left side will be taken up in the next slice down
        elif endingTime >= slice.startTime:
          # index of lowest timestamp that doesn't precede slice.startTime
          boundaryIndex = bisect_left(timestamps, slice.startTime)
          sequenceWithinSlice = sequence[boundaryIndex:]
          # write the leftovers on the next earlier slice
          sequence = sequence[:boundaryIndex]
          slice.write(sequenceWithinSlice)

        if not sequence:
          break

        sliceBoundary = slice.startTime

      else:  # slice list exhausted with stuff still to write
        needsEarlierSlice.append(sequence)

      if not slicesExist:
        sequences.append(sequence)
        needsEarlierSlice = sequences
        break

    for sequence in needsEarlierSlice:
      slice = CeresSlice.create(self, int(sequence[0][0]), self.timeStep)
      slice.write(sequence)
      self.clearSliceCache()

  def compact(self, datapoints):
    """Compacts datapoints into a list of contiguous, sorted lists of points with duplicate
timestamps and null values removed

      :param datapoints: List of datapoint tuples ``[(timestamp, value), ...]``

      :returns: A list of lists of contiguous sorted datapoint tuples
                ``[[(timestamp, value), ...], ...]``
    """
    datapoints = sorted(((int(timestamp), float(value))
                         for timestamp, value in datapoints if value is not None),
                        key=lambda datapoint: datapoint[0])
    sequences = []
    sequence = []
    minimumTimestamp = 0  # used to avoid duplicate intervals

    for timestamp, value in datapoints:
      timestamp -= timestamp % self.timeStep  # round it down to a proper interval

      if not sequence:
        sequence.append((timestamp, value))

      else:
        if timestamp == minimumTimestamp:  # overwrite duplicate intervals with latest value
          sequence[-1] = (timestamp, value)
          continue

        if timestamp == sequence[-1][0] + self.timeStep:  # append contiguous datapoints
          sequence.append((timestamp, value))

        else:  # start a new sequence if not contiguous
          sequences.append(sequence)
          sequence = [(timestamp, value)]

      minimumTimestamp = timestamp

    if sequence:
      sequences.append(sequence)

    return sequences


class CeresSlice(object):
  __slots__ = ('node', 'startTime', 'timeStep', 'fsPath')

  def __init__(self, node, startTime, timeStep):
    self.node = node
    self.startTime = startTime
    self.timeStep = timeStep
    self.fsPath = join(node.fsPath, '%d@%d.slice' % (startTime, timeStep))

  def __repr__(self):
    return "<CeresSlice[0x%x]: %s>" % (id(self), self.fsPath)
  __str__ = __repr__

  @property
  def isEmpty(self):
    return getsize(self.fsPath) == 0

  @property
  def endTime(self):
    return self.startTime + ((getsize(self.fsPath) // DATAPOINT_SIZE) * self.timeStep)

  @property
  def mtime(self):
    return getmtime(self.fsPath)

  @classmethod
  def create(cls, node, startTime, timeStep):
    slice = cls(node, startTime, timeStep)
    fileHandle = open(slice.fsPath, 'wb')
    fileHandle.close()
    os.chmod(slice.fsPath, SLICE_PERMS)
    return slice

  def read(self, fromTime, untilTime):
    timeOffset = int(fromTime) - self.startTime

    if timeOffset < 0:
      raise InvalidRequest("requested time range (%d, %d) precedes this slice: %d" % (
          fromTime, untilTime, self.startTime))

    pointOffset = timeOffset // self.timeStep
    byteOffset = pointOffset * DATAPOINT_SIZE

    if byteOffset >= getsize(self.fsPath):
      raise NoData()

    with open(self.fsPath, 'rb') as fileHandle:
      fileHandle.seek(byteOffset)

      timeRange = int(untilTime - fromTime)
      pointRange = timeRange // self.timeStep
      byteRange = pointRange * DATAPOINT_SIZE
      packedValues = fileHandle.read(byteRange)

      pointsReturned = len(packedValues) // DATAPOINT_SIZE
      format = '!' + ('d' * pointsReturned)
      values = struct.unpack(format, packedValues)
      values = [v if not isnan(v) else None for v in values]

      endTime = fromTime + (len(values) * self.timeStep)
      # print '[DEBUG slice.read] startTime=%s fromTime=%s untilTime=%s' % (
      #    self.startTime, fromTime, untilTime)
      # print '[DEBUG slice.read] timeInfo = (%s, %s, %s)' % (fromTime, endTime, self.timeStep)
      # print '[DEBUG slice.read] values = %s' % str(values)
      return TimeSeriesData(fromTime, endTime, self.timeStep, values)

  def write(self, sequence):
    beginningTime = sequence[0][0]
    timeOffset = beginningTime - self.startTime
    pointOffset = timeOffset // self.timeStep
    byteOffset = pointOffset * DATAPOINT_SIZE

    values = [v for t, v in sequence]
    format = '!' + ('d' * len(values))
    packedValues = struct.pack(format, *values)

    try:
      filesize = getsize(self.fsPath)
    except OSError as e:
      if e.errno == errno.ENOENT:
        raise SliceDeleted()
      else:
        raise

    byteGap = byteOffset - filesize
    if byteGap > 0:  # pad the allowable gap with nan's
      pointGap = byteGap // DATAPOINT_SIZE
      if pointGap > MAX_SLICE_GAP:
        raise SliceGapTooLarge()
      else:
        packedGap = PACKED_NAN * pointGap
        packedValues = packedGap + packedValues
        byteOffset -= byteGap

    with open(self.fsPath, 'r+b') as fileHandle:
      if LOCK_WRITES:
        fcntl.flock(fileHandle.fileno(), fcntl.LOCK_EX)
      try:
        fileHandle.seek(byteOffset)
      except IOError:
        # print " IOError: fsPath=%s byteOffset=%d size=%d sequence=%s" % (
        #   self.fsPath, byteOffset, filesize, sequence)
        raise
      fileHandle.write(packedValues)

  def deleteBefore(self, t):
    if not exists(self.fsPath):
      raise SliceDeleted()

    if t % self.timeStep != 0:
      t = t - (t % self.timeStep) + self.timeStep
    timeOffset = t - self.startTime
    if timeOffset < 0:
      return

    pointOffset = timeOffset // self.timeStep
    byteOffset = pointOffset * DATAPOINT_SIZE
    if not byteOffset:
      return

    self.node.clearSliceCache()
    with open(self.fsPath, 'r+b') as fileHandle:
      if LOCK_WRITES:
        fcntl.flock(fileHandle.fileno(), fcntl.LOCK_EX)
      fileHandle.seek(byteOffset)
      fileData = fileHandle.read()
      if fileData:
        fileHandle.seek(0)
        fileHandle.write(fileData)
        fileHandle.truncate()
        fileHandle.close()
        newFsPath = join(dirname(self.fsPath), "%d@%d.slice" % (t, self.timeStep))
        os.rename(self.fsPath, newFsPath)
      else:
        os.unlink(self.fsPath)
        raise SliceDeleted()

  def __lt__(self, other):
    return self.startTime < other.startTime


class TimeSeriesData(object):
  __slots__ = ('startTime', 'endTime', 'timeStep', 'values')

  def __init__(self, startTime, endTime, timeStep, values):
    self.startTime = startTime
    self.endTime = endTime
    self.timeStep = timeStep
    self.values = values

  @property
  def timestamps(self):
    return range(self.startTime, self.endTime, self.timeStep)

  def __iter__(self):
    return izip(self.timestamps, self.values)

  def __len__(self):
    return len(self.values)

  def merge(self, other):
    for timestamp, value in other:
      if value is None:
        continue

      timestamp -= timestamp % self.timeStep
      if timestamp < self.startTime:
        continue

      index = int((timestamp - self.startTime) // self.timeStep)

      try:
        if self.values[index] is None:
          self.values[index] = value
      except IndexError:
        continue


class CorruptNode(Exception):
  def __init__(self, node, problem):
    Exception.__init__(self, problem)
    self.node = node
    self.problem = problem


class NoData(Exception):
  pass


class NodeNotFound(Exception):
  pass


class NodeDeleted(Exception):
  pass


class InvalidRequest(Exception):
  pass


class InvalidAggregationMethod(Exception):
  pass


class SliceGapTooLarge(Exception):
  "For internal use only"


class SliceDeleted(Exception):
  pass


def aggregate(aggregationMethod, values):
  # Filter out None values
  knownValues = list(filter(lambda x: x is not None, values))
  if len(knownValues) is 0:
    return None
  # Aggregate based on method
  if aggregationMethod == 'average':
    return float(sum(knownValues)) / float(len(knownValues))
  elif aggregationMethod == 'sum':
    return float(sum(knownValues))
  elif aggregationMethod == 'last':
    return knownValues[-1]
  elif aggregationMethod == 'max':
    return max(knownValues)
  elif aggregationMethod == 'min':
    return min(knownValues)
  else:
    raise InvalidAggregationMethod("Unrecognized aggregation method %s" %
                                   aggregationMethod)


def aggregateSeries(method, oldTimeStep, newTimeStep, values):
  # Aggregate current values to fit newTimeStep.
  # Makes the assumption that the caller has already guaranteed
  # that newTimeStep is bigger than oldTimeStep.
  factor = int(newTimeStep // oldTimeStep)
  newValues = []
  subArr = []
  for val in values:
    subArr.append(val)
    if len(subArr) == factor:
      newValues.append(aggregate(method, subArr))
      subArr = []

  if len(subArr):
    newValues.append(aggregate(method, subArr))

  return newValues


def getTree(path):
  while path not in (os.sep, ''):
    if isdir(join(path, '.ceres-tree')):
      return CeresTree(path)

    path = dirname(path)


def setDefaultNodeCachingBehavior(behavior):
  global DEFAULT_NODE_CACHING_BEHAVIOR

  behavior = behavior.lower()
  if behavior not in ('none', 'all'):
    raise ValueError("invalid caching behavior '%s'" % behavior)

  DEFAULT_NODE_CACHING_BEHAVIOR = behavior


def setDefaultSliceCachingBehavior(behavior):
  global DEFAULT_SLICE_CACHING_BEHAVIOR

  behavior = behavior.lower()
  if behavior not in ('none', 'all', 'latest'):
    raise ValueError("invalid caching behavior '%s'" % behavior)

  DEFAULT_SLICE_CACHING_BEHAVIOR = behavior

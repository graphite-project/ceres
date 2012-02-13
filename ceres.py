# Ceres requires Python 2.6 or newer
import os
import time
import struct
import json
import errno
from math import isnan
from itertools import izip
from os.path import isdir, exists, join, basename, dirname, abspath, getsize, getmtime
from glob import glob
from bisect import bisect_left


TIMESTAMP_FORMAT = "!L"
TIMESTAMP_SIZE = struct.calcsize(TIMESTAMP_FORMAT)
DATAPOINT_FORMAT = "!d"
DATAPOINT_SIZE = struct.calcsize(DATAPOINT_FORMAT)
NAN = float('nan')
PACKED_NAN = struct.pack(DATAPOINT_FORMAT, NAN)
MAX_SLICE_GAP = 80
DEFAULT_TIMESTEP = 60
DEFAULT_SLICE_CACHING_BEHAVIOR = 'none'
SLICE_PERMS = 0644


class CeresTree:
  def __init__(self, root):
    if isdir(root):
      self.root = abspath(root)
    else:
      raise ValueError("Invalid root directory '%s'" % root)
    self.nodeCache = {}


  def __repr__(self):
    return "<CeresTree[0x%x]: %s>" % (id(self), self.root)
  __str__ = __repr__


  @classmethod
  def createTree(cls, root, **props):
    ceresDir = join(root, '.ceres-tree')
    if not isdir(ceresDir):
      os.system("mkdir -p '%s'" % ceresDir)

    for prop,value in props.items():
      propFile = join(ceresDir, prop)
      fh = open(propFile, 'w')
      fh.write( str(value) )
      fh.close()

    return cls(root)


  def walk(self, **kwargs):
    for (fsPath, subdirs, filenames) in os.walk(self.root, **kwargs):
      if CeresNode.isNodeDir(fsPath):
        nodePath = self.getNodePath(fsPath)
        yield CeresNode(self, nodePath, fsPath)


  def getFilesystemPath(self, nodePath):
    return join(self.root, nodePath.replace('.', '/'))


  def getNodePath(self, fsPath):
    fsPath = abspath(fsPath)
    if not fsPath.startswith(self.root):
      raise ValueError("path '%s' not beneath tree root '%s'" % (fsPath, self.root))

    nodePath = fsPath[ len(self.root): ].strip('/').replace('/', '.')
    return nodePath


  def hasNode(self, nodePath):
    return isdir( self.getFilesystemPath(nodePath) )


  def getNode(self, nodePath):
    if nodePath not in self.nodeCache:
      fsPath = self.getFilesystemPath(nodePath)
      if CeresNode.isNodeDir(fsPath):
        self.nodeCache[nodePath] = CeresNode(self, nodePath, fsPath)
      else:
        return None

    return self.nodeCache[nodePath]


  def find(self, nodePattern, fromTime=None, untilTime=None):
    for fsPath in glob( self.getFilesystemPath(nodePattern) ):
      if CeresNode.isNodeDir(fsPath):
        nodePath = self.getNodePath(fsPath)
        node = self.getNode(nodePath)

        if fromTime is None and untilTime is None:
          yield node
        elif node.hasDataForInterval(fromTime, untilTime):
          yield node


  def createNode(self, nodePath, **properties):
    return CeresNode.create(self, nodePath, **properties)


  def store(self, nodePath, datapoints):
    node = self.getNode(nodePath)

    if node is None:
      raise NodeNotFound("The node '%s' does not exist in this tree" % nodePath)

    node.write(datapoints)


  def fetch(self, nodePath, fromTime, untilTime):
    node = self.getNode(nodePath)

    if not node:
      raise NodeNotFound("the node '%s' does not exist in this tree" % nodePath)

    return node.read(fromTime, untilTime)



class CeresNode(object):
  __slots__ = ('tree', 'nodePath', 'fsPath',
               'metadataFile', 'timeStep',
               'sliceCache', 'sliceCachingBehavior')

  def __init__(self, tree, nodePath, fsPath):
    self.tree = tree
    self.nodePath = nodePath
    self.fsPath = fsPath
    self.metadataFile = join(fsPath, '.ceres-node')
    self.timeStep = None
    self.sliceCache = None
    self.sliceCachingBehavior = DEFAULT_SLICE_CACHING_BEHAVIOR


  def __repr__(self):
    return "<CeresNode[0x%x]: %s>" % (id(self), self.nodePath)
  __str__ = __repr__


  @classmethod
  def create(cls, tree, nodePath, **properties):
    # Create the node directory
    fsPath = tree.getFilesystemPath(nodePath)
    os.system("mkdir -p '%s'" % fsPath)

    # Create the initial metadata
    timeStep = properties['timeStep'] = properties.get('timeStep', DEFAULT_TIMESTEP)
    node = cls(tree, nodePath, fsPath)
    node.writeMetadata(properties)

    # Create the initial data file
    #now = int( time.time() )
    #baseTime = now - (now % timeStep)
    #slice = CeresSlice.create(node, baseTime, timeStep)

    return node


  @staticmethod
  def isNodeDir(path):
    return isdir(path) and exists( join(path, '.ceres-node') )


  @classmethod
  def fromFilesystemPath(cls, fsPath):
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
    return [ (slice.startTime, slice.endTime, slice.timeStep) for slice in self.slices ]


  def readMetadata(self):
    metadata = json.load( open(self.metadataFile, 'r') )
    self.timeStep = int( metadata['timeStep'] )
    return metadata


  def writeMetadata(self, metadata):
    self.timeStep = int( metadata['timeStep'] )

    f = open(self.metadataFile, 'w')
    json.dump(metadata, f)
    f.close()


  @property
  def slices(self):
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
        self.sliceCache = [ CeresSlice(self, *info) for info in self.readSlices() ]
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
    if not exists(self.fsPath):
      raise NodeDeleted()

    slice_info = []
    for filename in os.listdir(self.fsPath):
      if filename.endswith('.slice'):
        startTime, timeStep = filename[:-6].split('@')
        slice_info.append( (int(startTime), int(timeStep)) )

    slice_info.sort(reverse=True)
    return slice_info


  def setSliceCachingBehavior(self, behavior):
    behavior = behavior.lower()
    if behavior not in ('none', 'all', 'latest'):
      raise ValueError("invalid caching behavior '%s'" % behavior)

    self.sliceCachingBehavior = behavior
    self.sliceCache = None


  def clearSliceCache(self):
    self.sliceCache = None


  def hasDataForInterval(self, fromTime, untilTime):
    slices = list(self.slices)
    if not slices:
      return False

    earliestData = slices[-1].startTime
    latestData = slices[0].endTime

    return ( (fromTime is None) or (fromTime < latestData) ) and \
           ( (untilTime is None) or (untilTime > earliestData) )


  def read(self, fromTime, untilTime):
    if self.timeStep is None:
      self.readMetadata()

    # Normalize the timestamps to fit proper intervals
    fromTime  = int( fromTime - (fromTime % self.timeStep) + self.timeStep )
    untilTime = int( untilTime - (untilTime % self.timeStep) + self.timeStep )

    sliceBoundary = None # to know when to split up queries across slices
    resultValues = []
    earliestData = None

    for slice in self.slices:
      # if the requested interval starts after the start of this slice
      if fromTime >= slice.startTime:
        try:
          series = slice.read(fromTime, untilTime)
        except NoData:
          break

        earliestData = series.startTime

        rightMissing = (untilTime - series.endTime) / self.timeStep
        rightNulls   = [ None for i in range(rightMissing - len(resultValues)) ]
        resultValues = series.values + rightNulls + resultValues
        break

      # or if slice contains data for part of the requested interval
      elif untilTime >= slice.startTime:
        # Split the request up if it straddles a slice boundary
        if (sliceBoundary is not None) and untilTime > sliceBoundary:
          requestUntilTime = sliceBoundary
        else:
          requestUntilTime = untilTime

        try:
          series = slice.read(slice.startTime, requestUntilTime)
        except NoData:
          continue

        earliestData = series.startTime

        rightMissing = (requestUntilTime - series.endTime) / self.timeStep
        rightNulls   = [ None for i in range(rightMissing) ]
        resultValues = series.values + rightNulls + resultValues

      # this is the right-side boundary on the next iteration
      sliceBoundary = slice.startTime

    # The end of the requested interval predates all slices
    if earliestData is None:
      missing = int(untilTime - fromTime) / self.timeStep
      resultValues = [ None for i in range(missing) ]

    # Left pad nulls if the start of the requested interval predates all slices
    else:
      leftMissing = (earliestData - fromTime) / self.timeStep
      leftNulls = [ None for i in range(leftMissing) ]
      resultValues = leftNulls + resultValues

    return TimeSeriesData(fromTime, untilTime, self.timeStep, resultValues)


  def write(self, datapoints):
    if self.timeStep is None:
      self.readMetadata()

    if not datapoints:
      return

    sequences = self.compact(datapoints)
    needsEarlierSlice = [] # keep track of sequences that precede all existing slices

    while sequences:
      sequence = sequences.pop()
      timestamps = [ t for t,v in sequence ]
      beginningTime = timestamps[0]
      endingTime = timestamps[-1]
      sliceBoundary = None # used to prevent writing sequences across slice boundaries
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
            boundaryIndex = bisect_left(timestamps, sliceBoundary) # index of highest timestamp that doesn't exceed sliceBoundary
            sequenceWithinSlice = sequence[:boundaryIndex]

          try:
            slice.write(sequenceWithinSlice)
          except SliceGapTooLarge:
            newSlice = CeresSlice.create(self, beginningTime, slice.timeStep)
            newSlice.write(sequenceWithinSlice)
            self.sliceCache = None
          except SliceDeleted:
            self.sliceCache = None
            self.write(datapoints) # recurse to retry
            return

          break

        elif endingTime >= slice.startTime: # sequence straddles the current slice, write the right side
          boundaryIndex = bisect_left(timestamps, slice.startTime) # index of lowest timestamp that doesn't preceed slice.startTime
          sequenceWithinSlice = sequence[boundaryIndex:]
          leftover = sequence[:boundaryIndex]
          sequences.append(leftover)
          slice.write(sequenceWithinSlice)

        else:
          needsEarlierSlice.append(sequence)

        sliceBoundary = slice.startTime

      if not slicesExist:
        sequences.append(sequence)
        needsEarlierSlice = sequences
        break

    for sequence in needsEarlierSlice:
      slice = CeresSlice.create(self, int(sequence[0][0]), self.timeStep)
      slice.write(sequence)
      self.sliceCache = None


  def compact(self, datapoints):
    datapoints = sorted( (int(timestamp), float(value))
                         for timestamp, value in datapoints
                         if value is not None )
    sequences = []
    sequence = []
    minimumTimestamp = 0 # used to avoid duplicate intervals

    for timestamp, value in datapoints:
      timestamp -= timestamp % self.timeStep # round it down to a proper interval

      if not sequence:
        sequence.append( (timestamp, value) )

      else:
        if not timestamp > minimumTimestamp: #drop duplicate intervals
          continue

        if timestamp == sequence[-1][0] + self.timeStep: # append contiguous datapoints
          sequence.append( (timestamp, value) )

        else: # start a new sequence if not contiguous
          sequences.append(sequence)
          sequence = [ (timestamp, value) ]

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
    return self.startTime + ((getsize(self.fsPath) / DATAPOINT_SIZE) * self.timeStep)


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
    timeOffset  = int(fromTime) - self.startTime

    if timeOffset < 0:
      raise InvalidRequest("requested time range (%d, %d) preceeds this slice: %d" % (fromTime, untilTime, self.startTime))

    pointOffset = timeOffset / self.timeStep
    byteOffset  = pointOffset * DATAPOINT_SIZE

    if byteOffset >= getsize(self.fsPath):
      raise NoData()

    fileHandle = open(self.fsPath, 'rb')
    fileHandle.seek(byteOffset)

    timeRange  = int(untilTime - fromTime)
    pointRange = timeRange / self.timeStep
    byteRange  = pointRange * DATAPOINT_SIZE
    packedValues = fileHandle.read(byteRange)

    pointsReturned = len(packedValues) / DATAPOINT_SIZE
    format = '!' + ('d' * pointsReturned)
    values = struct.unpack(format, packedValues)
    values = [ v if not isnan(v) else None for v in values ]

    endTime = fromTime + (len(values) * self.timeStep)
    #print '[DEBUG slice.read] startTime=%s fromTime=%s untilTime=%s' % (self.startTime, fromTime, untilTime)
    #print '[DEBUG slice.read] timeInfo = (%s, %s, %s)' % (fromTime, endTime, self.timeStep)
    #print '[DEBUG slice.read] values = %s' % str(values)
    return TimeSeriesData(fromTime, endTime, self.timeStep, values)


  def write(self, sequence):
    beginningTime = sequence[0][0]
    timeOffset  = beginningTime - self.startTime
    pointOffset = timeOffset / self.timeStep
    byteOffset  = pointOffset * DATAPOINT_SIZE

    values = [ v for t,v in sequence ]
    format = '!' + ('d' * len(values))
    packedValues = struct.pack(format, *values)

    try:
      filesize = getsize(self.fsPath)
    except OSError, e:
      if e.errno == errno.ENOENT:
        raise SliceDeleted()
      else:
        raise

    byteGap = byteOffset - filesize
    if byteGap > 0: # pad the allowable gap with nan's

      if byteGap > MAX_SLICE_GAP:
        raise SliceGapTooLarge()
      else:
        pointGap = byteGap / DATAPOINT_SIZE
        packedGap = PACKED_NAN * pointGap
        packedValues = packedGap + packedValues
        byteOffset -= byteGap

    with file(self.fsPath, 'r+b') as fileHandle:
      try:
        fileHandle.seek(byteOffset)
      except IOError:
        print " IOError: fsPath=%s byteOffset=%d size=%d sequence=%s" % (self.fsPath, byteOffset, filesize, sequence)
        raise
      fileHandle.write(packedValues)


  def deleteBefore(self, t):
    if not exists(self.fsPath):
      raise SliceDeleted()

    t = t - (t % self.timeStep)
    timeOffset = t - self.startTime
    if timeOffset < 0:
      return

    pointOffset = timeOffset / self.timeStep
    byteOffset  = pointOffset * DATAPOINT_SIZE
    if not byteOffset:
      return

    self.node.clearSliceCache()
    with file(self.fsPath, 'r+b') as fileHandle:
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


  def __cmp__(self, other):
    return cmp(self.startTime, other.startTime)



class TimeSeriesData(object):
  __slots__ = ('startTime', 'endTime', 'timeStep', 'values')

  def __init__(self, startTime, endTime, timeStep, values):
    self.startTime = startTime
    self.endTime = endTime
    self.timeStep = timeStep
    self.values = values

  @property
  def timestamps(self):
    return xrange(self.startTime, self.endTime, self.timeStep)

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

      index = int( (timestamp - self.startTime) / self.timeStep )

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


class SliceGapTooLarge(Exception):
  "For internal use only"

class SliceDeleted(Exception):
  pass


def getTree(path):
  while path not in ('/', ''):
    if isdir( join(path, '.ceres-tree') ):
      return CeresTree(path)

    path = dirname(path)


def setDefaultSliceCachingBehavior(behavior):
  global DEFAULT_SLICE_CACHING_BEHAVIOR

  behavior = behavior.lower()
  if behavior not in ('none', 'all', 'latest'):
    raise ValueError("invalid caching behavior '%s'" % behavior)

  DEFAULT_SLICE_CACHING_BEHAVIOR = behavior

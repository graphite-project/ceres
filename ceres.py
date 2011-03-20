import os
import time
import struct
import json
from math import isnan
from itertools import izip
from os.path import isdir, exists, join, basename, dirname, getsize, abspath
from glob import glob
from bisect import bisect_left


TIMESTAMP_FORMAT = "!L"
TIMESTAMP_SIZE = struct.calcsize(TIMESTAMP_FORMAT)
DATAPOINT_FORMAT = "!d"
DATAPOINT_SIZE = struct.calcsize(DATAPOINT_FORMAT)
NAN = float('nan')
PACKED_NAN = struct.pack(DATAPOINT_FORMAT, NAN)
MAX_SLICE_GAP = 0
DEFAULT_TIMESTEP = 60


class CeresTree:
  def __init__(self, root):
    if isdir(root):
      self.root = abspath(root)
    else:
      raise ValueError("Invalid root directory '%s'" % root)


  @classmethod
  def createTree(cls, root, **props):
    ceresDir = join(root, '.ceres-tree')
    if not isdir(ceresDir):
      os.makedirs(ceresDir)

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
    fsPath = self.getFilesystemPath(nodePath)

    if CeresNode.isNodeDir(fsPath):
      return CeresNode(self, nodePath, fsPath)


  def find(self, nodePattern, fromTime=None, untilTime=None):
    for fsPath in glob( self.getFilesystemPath(nodePattern) ):
      if CeresNode.isNodeDir(fsPath):
        nodePath = self.getNodePath(fsPath)
        node = CeresNode(self, nodePath, fsPath)

        if node.hasDataForInterval(fromTime, untilTime):
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



class CeresNode:
  __slots__ = ('metadata', 'slices', 'tree', 'nodePath', 'fsPath', 'metadataFile')
  metadata = None
  slices = None

  def __init__(self, tree, nodePath, fsPath):
    self.tree = tree
    self.nodePath = nodePath
    self.fsPath = fsPath
    self.metadataFile = join(fsPath, '.ceres-node')


  @classmethod
  def create(cls, tree, nodePath, **properties):
    # Create the node directory
    fsPath = tree.getFilesystemPath(nodePath)
    os.makedirs(fsPath)

    # Create the initial metadata
    timeStep = properties['timeStep'] = properties.get('timeStep', DEFAULT_TIMESTEP)
    node = cls(tree, nodePath, fsPath)
    node.writeMetadata(properties)

    # Create the initial data file
    now = int( time.time() )
    baseTime = now - (now % timeStep)
    slice = CeresSlice.create(node, baseTime, timeStep)

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
  def timeStep(self):
    if self.metadata is None:
      self.readMetadata()

    return self.metadata['timeStep']


  @property
  def size(self):
    if not self.slices:
      self.readSlices()
    return sum(slice.size for slice in self.slices)


  @property
  def last_updated(self):
    if not self.slices:
      self.readSlices()

    if self.slices:
      return self.slices[0].mtime
    else:
      return 0


  @property
  def slice_info(self):
    if not self.slices:
      self.readSlices()

    return [ (slice.startTime, slice.endTime, slice.timeStep) for slice in self.slices ]


  def readMetadata(self):
    if exists(self.metadataFile):
      self.metadata = json.load( open(self.metadataFile, 'r') )
    else:
      self.metadata = {}


  def writeMetadata(self, properties):
    if self.metadata is None:
      self.readMetadata()

    self.metadata.update(properties)
    f = open(self.metadataFile, 'w')
    json.dump(self.metadata, f)
    f.close()


  def readSlices(self):
    slices = []

    for filename in os.listdir(self.fsPath):
      if filename.endswith('.slice'):
        startTime, timeStep = filename[:-6].split('@')
        slices.append( CeresSlice(self, int(startTime), int(timeStep)) )

    slices.sort(reverse=True)
    self.slices = slices


  def hasDataForInterval(self, fromTime, untilTime):
    if not self.slices:
      self.readSlices()

      if not self.slices:
        raise CorruptNode(self, "No slices exist for node %s" % self.fsPath)

    earliestData = self.slices[-1].startTime
    latestData = self.slices[0].endTime

    return ( (fromTime is None) or (fromTime < latestData) ) and \
           ( (untilTime is None) or (untilTime > earliestData) )


  def read(self, fromTime, untilTime):
    if not self.slices:
      self.readSlices()

      if not self.slices:
        raise CorruptNode(self, "No slices exist for node %s" % self.fsPath)

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
    if not datapoints:
      return

    if not self.slices:
      self.readSlices()

      if not self.slices:
        raise CorruptNode(self, "No slices exist for node %s" % self.fsPath)

    sequences = self.compact(datapoints)
    needsEarlierSlice = [] # keep track of sequences that precede all existing slices

    while sequences:
      sequence = sequences.pop()
      timestamps = [ t for t,v in sequence ]
      beginningTime = timestamps[0]
      endingTime = timestamps[-1]
      sliceBoundary = None # used to prevent writing sequences across slice boundaries

      for slice in self.slices:

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
            self.slices.insert(self.slices.index(slice), newSlice)

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

    for sequence in needsEarlierSlice:
      slice = CeresSlice.create(self, int(sequence[0][0]), self.timeStep)
      slice.write(sequence)


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



class CeresSlice:
  __slots__ = ('node', 'startTime', 'timeStep', 'fsPath')

  def __init__(self, node, startTime, timeStep):
    self.node = node
    self.startTime = startTime
    self.timeStep = timeStep
    self.fsPath = join(node.fsPath, '%d@%d.slice' % (startTime, timeStep))


  @property
  def endTime(self):
    return self.startTime + ((getsize(self.fsPath) / DATAPOINT_SIZE) * self.timeStep)


  @property
  def size(self):
    return getsize(self.fsPath)


  @property
  def mtime(self):
    return os.stat(self.fsPath).st_mtime


  @classmethod
  def create(cls, node, startTime, timeStep):
    slice = cls(node, startTime, timeStep)
    fileHandle = open(slice.fsPath, 'wb')
    fileHandle.close()
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

    filesize = getsize(self.fsPath)
    byteGap = byteOffset - filesize
    if byteGap > 0: # if we're writing beyond the end of the file we pad the gap with nan's

      if filesize == 0: # if we're empty we simply rename the slice to reflect the initial timestamp
        newPath = join(self.node.fsPath, '%d@%d.slice' % (beginningTime, self.timeStep))
        os.rename(self.fsPath, newPath)
        self.fsPath = newPath
        self.startTime = beginningTime
        byteOffset = 0
        #self.node.slices.sort(reverse=True) #I don't think this is necessary

      elif byteGap > MAX_SLICE_GAP:
        raise SliceGapTooLarge()

      else:
        pointGap = byteGap / DATAPOINT_SIZE
        packedGap = PACKED_NAN * pointGap
        packedValues = packedGap + packedValues
        byteOffset -= byteGap

    with file(self.fsPath, 'r+b') as fileHandle:
      fileHandle.seek(byteOffset)
      fileHandle.write(packedValues)


  def __cmp__(self, other):
    return cmp(self.startTime, other.startTime)



class TimeSeriesData:
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


class InvalidRequest(Exception):
  pass


class SliceGapTooLarge(Exception):
  "For internal use only"


def getTree(path):
  while path not in ('/', ''):
    if isdir( join(path, '.ceres-tree') ):
      return CeresTree(path)

    path = dirname(path)

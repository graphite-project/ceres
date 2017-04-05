import ceres

try:
  if 'CERES_MAX_SLICE_GAP' in settings:
    ceres.MAX_SLICE_GAP = int(settings['CERES_MAX_SLICE_GAP'])
  if ceres.CAN_LOCK and 'CERES_LOCK_WRITES' in settings:
    ceres.LOCK_WRITES = int(settings['CERES_LOCK_WRITES'])
except KeyError:
  pass


# Roll-up slices on disk that don't match any retentions in metadata.
def node_found(node):
  metadata = node.readMetadata()

  if not node.slices:
    return

  retentions = dict(iter(metadata.get('retentions', [])))
  method = metadata.get('aggregationMethod', 'average')

  for slice in node.slices:
    found = filter(lambda x: x == slice.timeStep, retentions)
    if len(found) != 0:
      continue
    # Not found, recalculate to next precision up.
    bigger_timeSteps = sorted(filter(lambda x: x > slice.timeStep, retentions))
    if len(bigger_timeSteps) != 0:
      new_timeStep = bigger_timeSteps[0]
      series = slice.read(slice.startTime, slice.endTime)
      # Aggregate and normalize it to the new interval.
      series.values = ceres.aggregateSeries(method, series.timeStep, new_timeStep, series.values)
      series.timeStep = new_timeStep
      series.startTime = series.startTime - (series.startTime % new_timeStep)
      series.endTime = series.startTime + (len(series.values) * series.timeStep)
      # Replace all None values with NaNs.
      new_sequence = [(t, v if v is not None else ceres.NAN) for t, v in series]
      if len(new_sequence) != 0:
         new_slice = ceres.CeresSlice.create(node, new_sequence[0][0], new_timeStep)
         log("rewriting slice in new time step: %s -> %s" % (str(slice), str(new_slice)))
         new_slice.write(new_sequence)
      try:
        slice.deleteBefore(slice.endTime)
      except ceres.SliceDeleted:
        pass

  return

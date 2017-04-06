from itertools import izip

try:
  from carbon.storage import loadStorageSchemas, loadAggregationSchemas
  SCHEMAS = loadStorageSchemas()
  AGGREGATION_SCHEMAS = loadAggregationSchemas()
except ImportError:
  SCHEMAS = []
  AGGREGATION_SCHEMAS = []


def determine_metadata(metric):
  metadata = dict(timeStep=None, retentions=None, aggregationMethod=None, xFilesFactor=None)

  # Storage rules.
  for schema in SCHEMAS:
    if schema.matches(metric):
      metadata['retentions'] = [archive.getTuple() for archive in schema.archives]
      metadata['timeStep'] = metadata['retentions'][0][0]
      break

  # Aggregation rules.
  for schema in AGGREGATION_SCHEMAS:
    if schema.matches(metric):
      metadata['xFilesFactor'], metadata['aggregationMethod'] = schema.archives
      break

  # Validate all metadata was set.
  for k in metadata.keys():
    if metadata[k] is None:
      raise Exception("Couldn't determine metadata")

  return metadata


# Update metadata to match carbon schemas.
def node_found(node):
  metadata = node.readMetadata()
  write_metadata = 0

  if not node.slices:
    return

  try:
    new_metadata = determine_metadata(node.nodePath)
  except Exception:
    return

  # Work out whether any storage rules have changed.
  if len(metadata) != len(new_metadata):
    write_metadata = 1
  else:
    # Zip together the current and new retention points and compare.
    retentions = izip(metadata['retentions'], new_metadata['retentions'])
    for (old, new) in retentions:
      (precision, retention) = zip(old, new)
      # If the precision or retentions differ, update the metadata.
      if precision[0] != precision[1] or retention[0] != retention[1]:
        write_metadata = 1
        break

  # Maybe update the other metadata fields.
  if metadata['timeStep'] != new_metadata['timeStep']:
    write_metadata = 1

  if metadata['xFilesFactor'] != new_metadata['xFilesFactor']:
    write_metadata = 1

  if metadata['aggregationMethod'] != new_metadata['aggregationMethod']:
    write_metadata = 1

  # If any changes, write out the metadata now so the writers start using it.
  if write_metadata:
    log("updating metadata: %s" % str(node))
    node.writeMetadata(new_metadata)

  return

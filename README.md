# Ceres

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/e714322dad124c279d42b217a763bf6e)](https://www.codacy.com/app/graphite-project/ceres?utm_source=github.com&utm_medium=referral&utm_content=graphite-project/ceres&utm_campaign=badger)
[![Build Status](https://secure.travis-ci.org/graphite-project/ceres.png)](http://travis-ci.org/graphite-project/ceres)
[![FOSSA Status](https://app.fossa.io/api/projects/git%2Bhttps%3A%2F%2Fgithub.com%2Fgraphite-project%2Fceres.svg?type=shield)](https://app.fossa.io/projects/git%2Bhttps%3A%2F%2Fgithub.com%2Fgraphite-project%2Fceres?ref=badge_shield)

Ceres is *not actively maintained*.

Ceres is a component of [Graphite][] as one of the time-series storage options available for use.
Ceres provides a file format for incoming metrics to be persisted when received from the network.
See also [Whisper][]

[Graphite]: https://github.com/graphite-project
[Graphite Web]: https://github.com/graphite-project/graphite-web
[Carbon]: https://github.com/graphite-project/carbon
[Whisper]: https://github.com/graphite-project/whisper
[Ceres]: https://github.com/graphite-project/ceres

## Overview

Ceres is a time-series database format intended to replace [Whisper][] as the default storage
format for [Graphite][]. In contrast with Whisper, Ceres is not a fixed-size database and is
designed to better support sparse data of arbitrary fixed-size resolutions. This allows Graphite
to distribute individual time-series across multiple servers or mounts.

Expected features such as roll-up aggregation and data expiration are not provided by Ceres itself,
but instead are implemented as maintenance plugins in [Carbon][].

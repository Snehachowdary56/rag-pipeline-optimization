# NorthLight Remote Production Gateway

Made-up sample manual for a demo. Not a real product.

## Overview

The remote production gateway sends live video between a venue and a central control room over IP networks. It encodes up to four channels of video with low delay.

## Network

The gateway sends video as SRT streams over UDP. It needs at least 20 megabits per second of upload bandwidth per HD channel. Packet loss recovery adds a buffer that is set to four times the round-trip time.

## Latency

End-to-end delay is usually between 150 and 400 milliseconds depending on the network. Lower buffer values reduce delay but make the stream more sensitive to packet loss.

## Security

Streams can be encrypted with AES-256 using a shared passphrase. Each gateway has its own administrator account, and the web interface should only be reached over HTTPS.

## Monitoring

The gateway dashboard shows bitrate, round-trip time and packet loss for each stream. An alarm is raised if packet loss stays above 2 percent for more than ten seconds.

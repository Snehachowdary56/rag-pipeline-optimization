# NorthLight Canvas Graphics Engine

Made-up sample manual for a demo. Not a real product.

## Overview

Canvas is a real-time graphics engine for on-air graphics such as lower thirds, scoreboards and full-screen data. Templates are built in the Canvas designer and played out by the Canvas server.

## Templates

A template is a reusable graphic with text fields, images and animations. Data fields in a template can be filled by an operator or linked to a live data source. Templates are saved in the project library.

## Data Sources

Canvas can read live data from CSV files, databases, REST APIs and sports scoring feeds. Data refreshes every second by default. A data source that stops updating turns amber in the data monitor.

## Playout

Graphics are played out on two independent channels with fill and key outputs. Each channel can preview the next graphic before it goes to air. The take command moves the previewed graphic to air.

## Performance

Canvas renders at the frame rate of the output format. Large textures should be kept under 4096 by 4096 pixels. If the render time goes above one frame, the performance meter turns red and frames may drop.

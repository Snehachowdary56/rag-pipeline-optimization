# NorthLight Matrix Video Router

Made-up sample manual for a demo. Not a real product.

## Overview

The Matrix router connects any input to any output. The largest frame is 288 inputs by 288 outputs. Routers in a facility can be controlled from hardware panels or software panels.

## Salvos

A salvo is a group of routes taken together with one button. Salvos are used to set up a studio for a show in a single step. Up to 1000 salvos can be stored.

## Locks and Protects

A lock stops anyone changing a destination. A protect stops other users changing it but lets the owner change it. Locks are used on transmission outputs during live shows.

## Control Protocol

Third-party systems control the router over TCP using the router control protocol on port 52116. Each command routes one source to one destination and the router replies with the new state.

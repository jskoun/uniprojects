# Uni Projects

Collection of small projects developed for various subjects taught during my undergrad. Also check my MPU6050 IMU [implementation](https://github.com/jskoun/mpu6050-arduino-orientation).

More info on implementations and scope are included under each project's folder.

## Marine Pollution Detection

This project used real world data from sources across the Adriatic region with the goal of creating a monitoring system for marine pollution. It was an early attempt to toy with statistics based classification, prediction and other time-series specific tasks. No machine learning techniques were utilized (at least, not in the modern sense of NNs etc).

The main idea was to start with the clean up of raw sensor data, then to perform statistical analysis to find relationships between variables (mostly uncovering correllation) and eliminate outliers. By gathering these findings the second step is to train models to perform these tasks automatically and label anomalies and outliers in real-time when deployed. Lastly, the prediction part would be able to make assesments of future values, in which case when paired with anomaly detection it could (in theory) provide an early warning of pollution.

## Ship Drawings Digitizer

The provided script was used during my internship to provide a quick way to digitize. It works in conjuction with [WebPlotDigitizer](https://github.com/automeris-io/WebPlotDigitizer). Shoutout to the creators and maintainers of this software as it has proven to be an amazing tool throughout my studies. Given a lines plan of a marine craft, the Naval Architect can then select points using different coordinate systems for each curve, and then export the data as a JSON file. This file is then parsed by the script, which saves the ship's form and can either export it to a NAPA (naval architecture sofware) readable format, or plot it directly in 3D to easily assess any issues.

## Taylor Vortex Interaction Simulation

For this assignement we were tasked with writing the software for solving and visualizing a 2D flow field given specific initial and boundary conditions. This was a project mainly focused on scientific computing, which gave me an insight on the importance of understanding numerical methods and how they tie with the algorithm's speed (and by extension, your software's speed).






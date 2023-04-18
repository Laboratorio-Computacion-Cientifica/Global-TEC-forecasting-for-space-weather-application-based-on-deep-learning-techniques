#!/usr/bin/env python
"""TEC configuration file.

This file has all the configuration variables for run TEC's scripts.

# URL sources for download TEC files
# Source 1 (International GNSS Service -IGS):
                         ftp://igs.ensg.ign.fr/pub/igs/products/ionosphere/
# Source 2 (The Crustal Dynamics Data Information System -CDDIS):
                         ftps://gdc.cddis.eosdis.nasa.gov/gp
"""

ftpServer: str = 'gdc.cddis.eosdis.nasa.gov/'
# ftpServer: str = 'igs.ensg.ign.fr/pub/igs/products/ionosphere/'
ftpDirectory: str = 'gps/products/ionex/'
ftpCredencialEmail: str = 'jnamour@herrera.unt.edu.ar'
centre: str = 'igsg'  # 'jprg'
localDestinyDirectoryForIonexFiles: str = \
    '../data/tec/raw_tec'

# src: neural network based model for global Total Electron Content forecasting
stationsOfInterest = {"station1": [-85, -120], "station2": [-85, 0],
                      "station3": [-85, 120], "station4": [-50, -120],
                      "station5": [-50, 0], "station6": [-50, 120],
                      "station7": [-20, -120], "station8": [-20, 0],
                      "station9": [-20, 120], "station10": [20, -120],
                      "station11": [20, 0], "station12": [20, 120],
                      "station13": [50, -120], "station14": [50, 0],
                      "station15": [50, 120], "station16": [85, -120],
                      "station17": [85, 0], "station18": [85, 120]}

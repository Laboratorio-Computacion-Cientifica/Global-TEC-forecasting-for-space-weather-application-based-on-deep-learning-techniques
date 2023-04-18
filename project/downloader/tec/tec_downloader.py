#!/usr/bin/env python
"""Downloader module for TEC files.

#sources:
#https://github.com/daniestevez/jupyter_notebooks/blob/master/IONEX.ipynb
"""

import subprocess
import os


def ionex_filename(year, day, centre, zipped=True):
    return '{}{:03d}0.{:02d}i{}'.format(centre, day, year % 100, '.Z'
                                        if zipped else '')


def ionex_ftp_path(year, day, centre):
    return '{:04d}/{:03d}/{}'.format(year, day,
                                     ionex_filename(year, day, centre))


def ionex_local_path(year, day, centre, directory, zipped=False):
    return directory + '/' + ionex_filename(year, day, centre, zipped)


def download_ionex(year, day, ftpServer, directory, CredencialEmail, centre,
                   outputDir):
    # wget.download(ionex_ftp_path(year, day, centre), output_dir)
    filename = ionex_ftp_path(year, day, centre)
    absolutePathOfFilename = 'ftps://' + ftpServer + directory + '/' + filename
    os.system('wget -P' +
              outputDir +
              ' --ftp-user anonymous '
              '--ftp-password jnamour@herrera.unt.edu.ar ' +
              absolutePathOfFilename)
    # subprocess.call(['gzip', '-d', ionex_local_path(year,
    #                                                day, centre,
    #                                                outputDir,
    #                                                zipped=True)])

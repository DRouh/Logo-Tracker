#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import posixpath
from urlparse import urlsplit
from urllib import unquote
import time
from urllib import FancyURLopener
import urllib2
import urllib


def url2filename(url):
    """Return basename corresponding to url.

    >>> print(url2filename('http://example.com/path/to/file%C3%80?opt=1'))
    fileÀ
    >>> print(url2filename('http://example.com/slash%2fname')) # '/' in name
    Traceback (most recent call last):
    ...
    ValueError
    """
    urlpath = urlsplit(url).path
    basename = posixpath.basename(unquote(urlpath))
    if (os.path.basename(basename) != basename or
                unquote(posixpath.basename(urlpath)) != basename):
        raise ValueError  # reject '%2f' or 'dir%5Cbasename.ext' on Windows
    return basename


if __name__ == '__main__':
    fname = 'ImageNet Pepsi.txt'
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip('\n') for x in content]

    for i, url in enumerate(content):
        print "Downloading {0}/{1}".format(i + 1, len(content))
        try:
            local_filename, headers = urllib.urlretrieve(url, url2filename(url))
        except IOError:
            print "Can't download {0}. Skipped.".format(url)
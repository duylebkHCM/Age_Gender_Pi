#!/bin/bash
rawdatadir=../../data/UTK_Face/raw/
interdatadir=../../data/UTK_Face/interim/
mkdir -p $interdatadir
cd $rawdatadir
for f in *.tar.gz; do tar -xvf "$f"; done


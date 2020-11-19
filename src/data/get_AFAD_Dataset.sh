#!/bin/bash
rawdatadir=../../../Data/AFAD_Dataset/raw/
interdatadir=../../../Data/AFAD_Dataset/interim/
processeddir=../../../Data/AFAD_Dataset/processed/

mkdir -p $rawdatadir
mkdir -p $interdatadir
mkdir -p $processeddir

cd $rawdatadir
git clone https://github.com/afad-dataset/tarball.git

cd afad-dataset
bash restore.sh
tar -xvf AFAD-Full.tar.xz


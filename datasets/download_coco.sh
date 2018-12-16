#!/usr/bin/env bash

path=datasets/COCO
items=(annotations_trainval2017.zip train2017.zip val2017.zip test2017.zip)

mkdir $path

wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P $path
wget http://images.cocodataset.org/zips/train2017.zip -P $path
wget http://images.cocodataset.org/zips/val2017.zip -P $path
wget http://images.cocodataset.org/zips/test2017.zip -P $path

for item in ${items[@]}; do
    unzip $path/$item -d $path
    rm $path/$item
done

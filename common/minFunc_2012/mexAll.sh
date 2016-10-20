#! /bin/bash

echo "Compiling minFunc files..."

mex='mkoctfile --mex'

$mex -o minFunc/compiled/mcholC minFunc/mex/mcholC.c
$mex -o minFunc/compiled/lbfgsC minFunc/mex/lbfgsC.c
$mex -o minFunc/compiled/lbfgsAddC minFunc/mex/lbfgsAddC.c
$mex -o minFunc/compiled/lbfgsProdC minFunc/mex/lbfgsProdC.c


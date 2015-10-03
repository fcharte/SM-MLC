#!/bin/bash
mkdir RunMLClassifier
javac -cp lib/weka.jar:lib/mulan.jar -d . Main.java
jar cvfm RunMLClassifier.jar manifest.txt RunMLClassifier/Main.class
rm -r -f RunMLClassifier


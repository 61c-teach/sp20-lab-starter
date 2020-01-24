#!/bin/bash
rm -f git.tar
git log > git.log
tar -cf ./git.tar git.log
rm -f git.log
git add ./git.tar
git commit -m "Added the git log tar to your lab00 folder"
git push origin master

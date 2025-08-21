#!/bin/bash

# get-metadata.sh
# Use this script to execute get-metadata.py with the default arguments that find
# all metadata available from PhilPapers.
#  
# The output can also be finely or coarsely parsed into a CSV file using parse-metadata.py.

# Note: At some point between May and July of 2025, the OAI handler seems to have moved from https://philarchive.org/oai.pl to https://api.philpapers.org/oai.pl
# The script has been updated to use the new URL.

CURRENT_DATE=$(date +%Y-%m-%d)
python /opt/extra/avijit/projects/moralkg/data/scripts/phil-papers/get-metadata.py -l https://api.philpapers.org/oai.pl -o "/opt/extra/avijit/projects/moralkg/data/metadata/phil-papers/${CURRENT_DATE}.xml"
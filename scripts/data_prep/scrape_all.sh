#!/bin/bash

# Scrape Collection 1 (NLV)
echo "Starting scrape for Collection 1..."
python3 -u scripts/scrape_nom_foundation.py --url "http://lib.nomfoundation.org/collection/1/" &

# Scrape Collection 2 (Chùa Thắng Nghiêm)
echo "Starting scrape for Collection 2..."
python3 -u scripts/scrape_nom_foundation.py --url "http://lib.nomfoundation.org/collection/2/" &

# Scrape Collection 3 (Chùa Phổ Nhân)
echo "Starting scrape for Collection 3..."
python3 -u scripts/scrape_nom_foundation.py --url "http://lib.nomfoundation.org/collection/3/" &

wait
echo "All scrapes completed."


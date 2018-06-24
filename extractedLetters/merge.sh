awk 'NF && !x[$0]++' *.csv > all_letters.csv

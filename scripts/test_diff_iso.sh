echo "Model: $1"

python test_diff.py --checkpoint $1 --sigma 0 --datasets sr-geo-15,Google-15 --approxdiff VAR --schedule quadratic
python test_diff.py --checkpoint $1 --sigma 1.2 --datasets sr-geo-15,Google-15 --approxdiff VAR --schedule quadratic
python test_diff.py --checkpoint $1 --sigma 2.4 --datasets sr-geo-15,Google-15 --approxdiff VAR --schedule quadratic
python test_diff.py --checkpoint $1 --sigma 3.6 --datasets sr-geo-15,Google-15 --approxdiff VAR --schedule quadratic
python test_diff.py --checkpoint $1 --sigma 4.8 --datasets sr-geo-15,Google-15 --approxdiff VAR --schedule quadratic

python test_diff.py --checkpoint $1 --sigma 2.4 --datasets sr-geo-15 --gpu 0 --skip 1 --approxdiff VAR --schedule quadratic
python test_diff.py --checkpoint $1 --sigma 2.4 --datasets sr-geo-15 --gpu 0 --skip 10 --approxdiff VAR --schedule quadratic
python test_diff.py --checkpoint $1 --sigma 2.4 --datasets sr-geo-15 --gpu 0 --skip 25 --approxdiff VAR --schedule quadratic
python test_diff.py --checkpoint $1 --sigma 2.4 --datasets sr-geo-15 --gpu 0 --skip 100 --approxdiff VAR --schedule quadratic
python test_diff.py --checkpoint $1 --sigma 2.4 --datasets sr-geo-15 --gpu 0 --skip 200 --approxdiff VAR --schedule quadratic
python test_diff.py --checkpoint $1 --sigma 2.4 --datasets sr-geo-15 --gpu 0 --skip 500 --approxdiff VAR --schedule quadratic